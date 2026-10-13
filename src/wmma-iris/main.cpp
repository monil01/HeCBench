/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <cstdint>
#include <hip/hip_fp16.h>
typedef half fp16;
typedef float fp32;
#include "reference.h"
#include <iris/iris.hpp>

static void check(int code) {
    if(code != IRIS_SUCCESS) { std::cerr << "IRIS error " << code << "\n"; exit(2); }
}
template<typename T>
static void initialize(T* mat, uint32_t m, uint32_t n) {
    srand(m*n);
    for(uint32_t i=0; i<m; ++i) for(uint32_t j=0; j<n; ++j) {
        auto value=(i*n+j)%13;
        mat[i*n+j]=(value%3) ? -static_cast<T>(value) : static_cast<T>(value);
    }
}
int main(int argc, char** argv) {
    if(argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <implementation> <M> <N> <K> <repeat> <verify>\n";
        return 1;
    }
    uint32_t impl=atoi(argv[1]), m=atoi(argv[2]), n=atoi(argv[3]), k=atoi(argv[4]);
    int32_t repeat=atoi(argv[5]), verify=atoi(argv[6]);
    const uint32_t WMMA_M=16, WMMA_N=16, WMMA_K=16, TILE_M=64, TILE_N=64;
    if(impl>1 || m<16 || n<16 || k<16 || m%16 || n%16 || k%16 ||
       (impl==1 && (m<64 || n<64))) {
        std::cout << "Unsupported size!\n"; return 0;
    }
    std::cout << "Initializing host data...\n";
    std::vector<fp16> a(size_t(m)*k), b(size_t(k)*n);
    std::vector<fp32> c(size_t(m)*n), d(size_t(m)*n,std::numeric_limits<fp32>::signaling_NaN());
    initialize(a.data(),m,k); initialize(b.data(),k,n); initialize(c.data(),m,n);
    iris::Platform platform;
    check(platform.init(nullptr,nullptr,true));
    int device=-1, backend=-1, type=0;
    for(int i=0;i<platform.ndevs();++i) {
        check(iris_device_info(i,iris_backend,&backend,nullptr));
        if(backend==iris_hip) { device=i; break; }
    }
    if(device<0) { std::cerr << "No IRIS HIP device\n"; return 2; }
    char name[256]={}; check(iris_device_info(device,iris_name,name,nullptr));
    check(iris_device_info(device,iris_type,&type,nullptr));
    int wave=0;
    {
        iris::DMem info(&wave,sizeof(wave));
        iris::Task probe; void* p[]={&info}; int modes[]={iris_w}; size_t g=64,l=64;
        check(probe.kernel("wave_info",1,nullptr,&g,&l,1,p,modes));
        check(probe.flush_out(info)); check(probe.submit(device,nullptr,true));
    }
    std::cout << "IRIS backend=" << backend << " type=" << type << " device=" << name << " wave=" << wave << "\n";
    if(wave!=64) { std::cerr << "HIP backend currently requires wave64\n"; return 2; }
    {
        iris::DMem ma(a.data(),a.size()*sizeof(fp16)), mb(b.data(),b.size()*sizeof(fp16));
        iris::DMem mc(c.data(),c.size()*sizeof(fp32)), md(d.data(),d.size()*sizeof(fp32));
        uint32_t lda=k, ldb=k, ldc=n, ldd=n;
        fp32 alpha=0.5f, beta=2.0f;
        size_t local[]={impl==0 ? size_t(wave) : size_t(TILE_M/WMMA_M*wave), impl==0 ? size_t(1) : size_t(TILE_N/WMMA_N)};
        size_t global[]={local[0]*(impl==0 ? m/WMMA_M : m/TILE_M),local[1]*(impl==0 ? n/WMMA_N : n/TILE_N)};
        auto launch=[&](bool flush) {
            iris::Task task;
            void* params[]={&m,&n,&k,&ma,&mb,&mc,&md,&lda,&ldb,&ldc,&ldd,&alpha,&beta};
            int modes[]={sizeof(m),sizeof(n),sizeof(k),iris_r,iris_r,iris_r,iris_rw,sizeof(lda),sizeof(ldb),sizeof(ldc),sizeof(ldd),sizeof(alpha),sizeof(beta)};
            check(task.kernel(impl==0 ? "gemm_impl0" : "gemm_impl1",2,nullptr,global,local,13,params,modes));
            if(flush) check(task.flush_out(md));
            check(task.submit(device,nullptr,true));
        };
        std::cout << "Initializing device data...\nLaunching GEMM kernel...\n";
        for(int32_t i=0;i<30;++i) launch(verify && i==29);
        if(verify) {
            std::cout << "Validating result with reference...\n";
            std::vector<fp32> ref(size_t(m)*n,std::numeric_limits<fp32>::signaling_NaN());
            gemm_cpu_h(m,n,k,a.data(),b.data(),c.data(),ref.data(),lda,ldb,ldc,ldd,alpha,beta);
            compareEqual<fp32>(d.data(),ref.data(),m*n);
        } else std::cout << "Skip validating result with reference\n";
        auto start=std::chrono::steady_clock::now();
        for(int32_t i=0;i<repeat;++i) launch(false);
        auto end=std::chrono::steady_clock::now();
        auto time=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        double elapsedTimeMs=time*1e-6;
        auto gFlops=static_cast<double>(m)*n*(1.0+2.0*k)*1.0e-9;
        auto tFlopsPerSec=gFlops*repeat/elapsedTimeMs;
        std::cout << "BlkM, BlkN, BlkK, MatM, MatN, MatK, alpha, lda, ldb, beta, ldc, ldd, elapsedMs, Problem Size(GFlops), TFlops/s\n";
        std::cout << WMMA_M << ", " << WMMA_N << ", " << WMMA_K << ", " << m << ", " << n << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", " << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << gFlops << ", " << tFlopsPerSec << "\n";
    }
    check(platform.finalize());
    std::cout << "Finished!\n"; return 0;
}
