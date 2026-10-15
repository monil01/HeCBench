/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include "utils.h"
#include <iris/iris.hpp>

static void check(int code) { if(code!=IRIS_SUCCESS) {fprintf(stderr,"IRIS error%d\n",code);exit(2);} }
struct Input {
    std::vector<float> a,b,values,output;
    std::vector<int> rows,columns;
    Input(bool csr,int m,int k,int n,int nnz):
        a(size_t(m)*k),b(size_t(k)*n),values(nnz),output(size_t(m)*n),
        rows(csr?m+1:nnz),columns(nnz) {
        init_matrix(a.data(),m,k,nnz);
        if(csr) init_csr(rows.data(),values.data(),columns.data(),a.data(),m,k,nnz);
        else init_coo(rows.data(),values.data(),columns.data(),a.data(),m,k,nnz);
        init_matrix(b.data(),k,n,k*n);
    }
};
static void dump(const char* prefix,const char* format,const char* suffix,const std::vector<float>& values) {
    std::string name=std::string(prefix)+"."+format+suffix;
    FILE* f=fopen(name.c_str(),"wb");
    if(!f || fwrite(values.data(),sizeof(float),values.size(),f)!=values.size()) {perror("dump");exit(3);}
    fclose(f);
}
static void multiply(Input& input,bool csr,int m,int k,int n,int nnz,int repeat,int verify,int device) {
    const char* format=csr?"CSR":"COO";
    {
        iris::DMem mr(input.rows.data(),input.rows.size()*sizeof(int));
        iris::DMem mi(input.columns.data(),input.columns.size()*sizeof(int));
        iris::DMem mv(input.values.data(),input.values.size()*sizeof(float));
        iris::DMem mb(input.b.data(),input.b.size()*sizeof(float));
        iris::DMem mc(input.output.data(),input.output.size()*sizeof(float));
        size_t local=64,global=((size_t(m)*n+local-1)/local)*local;
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<repeat;++i) {
            iris::Task task;
            void* params[]={&mr,&mi,&mv,&mb,&mc,&m,&n,&nnz};
            int modes[]={iris_r,iris_r,iris_r,iris_r,iris_w,sizeof(m),sizeof(n),sizeof(nnz)};
            check(task.kernel(csr?"spmm_csr":"spmm_coo",1,nullptr,&global,&local,8,params,modes));
            if(i==repeat-1 && (verify || getenv("HECBENCH_DUMP_PREFIX"))) check(task.flush_out(mc));
            check(task.submit(device,nullptr,true));
        }
        auto end=std::chrono::steady_clock::now();
        auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        printf("Average execution time of SPGEMM (%s) compute: %f (us)\n",format,(ns*1e-3f)/repeat);
    }
    if(verify || getenv("HECBENCH_DUMP_PREFIX")) {
        std::vector<float> reference(input.output.size());
        gemm(input.a.data(),input.b.data(),reference.data(),k,m,n);
        if(verify) {
            printf("Computing the reference SPGEMM results..\n");
            bool correct=true;
            for(size_t i=0;i<reference.size();++i)
                if(fabsf(input.output[i]-reference[i])>1e-2f) {printf("@%zu %f != %f\n",i,input.output[i],reference[i]);correct=false;break;}
            printf("spgemm_example test %s\n",correct?"PASSED":"FAILED: wrong result");
        }
        if(const char* prefix=getenv("HECBENCH_DUMP_PREFIX")) {
            dump(prefix,format,".bin",input.output); dump(prefix,format,".expected.bin",reference);
        }
    }
}
int main(int argc,char** argv) {
    if(argc!=7) {printf("Usage %s <M> <K> <N> <A_nnz> <repeat> <verify>\n",argv[0]);return 1;}
    int m=atoi(argv[1]),k=atoi(argv[2]),n=atoi(argv[3]),nnz=atoi(argv[4]),repeat=atoi(argv[5]),verify=atoi(argv[6]);
    if(m<=0 || k<=0 || n<=0 || nnz<=0 || nnz>size_t(m)*k || repeat<=0) return 1;
    // Preserve the unseeded drand48 sequence across both source paths before init.
    Input coo(false,m,k,n,nnz),csr(true,m,k,n,nnz);
    iris::Platform platform;check(platform.init(nullptr,nullptr,true));
    int device=-1,backend=-1,type=0;
    for(int i=0;i<platform.ndevs();++i) {check(iris_device_info(i,iris_backend,&backend,nullptr));if(backend==iris_hip){device=i;break;}}
    if(device<0){fprintf(stderr,"No IRIS HIP GPU\n");return 2;}
    char name[256]={};check(iris_device_info(device,iris_name,name,nullptr));check(iris_device_info(device,iris_type,&type,nullptr));
    printf("IRIS backend=%d type=%d device=%s\n",backend,type,name);
    multiply(coo,false,m,k,n,nnz,repeat,verify,device);
    multiply(csr,true,m,k,n,nnz,repeat,verify,device);
    check(platform.finalize());return 0;
}
