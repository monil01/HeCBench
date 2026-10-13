#include <iris/iris.hpp>
#include <hip/hip_fp16.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
static void solve(unsigned n,unsigned repeat,unsigned bx,unsigned by,float sf,float zf,bool int4) {
  assert(by<=64 && bx*by<=1024 && n/bx >= (int4 ? 16u : 8u));
  assert(n%by==0);
  std::mt19937 gen(19937);
  std::uniform_int_distribution<int> dis(int4 ? 0 : -128,int4 ? 15 : 127);
  std::vector<uint8_t> mat(size_t(n)*n/(int4 ? 2 : 1));
  for (auto& m : mat) {
    if (int4) { int x=dis(gen),y=dis(gen); m=x|(y<<4); }
    else m=static_cast<int8_t>(dis(gen));
  }
  gen.seed(19937); std::uniform_real_distribution<float> real(0,1);
  std::vector<__half> vec(n),result(n);
  for (auto& x : vec) x=__float2half(real(gen));
  __half scale=__float2half(sf),zero=__float2half(zf);
  unsigned num_per_thread=n/bx;
  iris::DMem mm(mat.data(),mat.size()),mv(vec.data(),n*sizeof(__half)),mr(result.data(),n*sizeof(__half));
  size_t gws[]={bx,n},lws[]={bx,by};
  auto start=std::chrono::steady_clock::now();
  for (unsigned r=0; r<repeat; ++r) {
    iris::Task task;
    void* args[]={&mm,&mv,&mr,&n,&scale,&zero,&num_per_thread};
    int info[]={iris_r,iris_r,iris_w,sizeof(n),sizeof(scale),sizeof(zero),sizeof(num_per_thread)};
    if (task.kernel(int4 ? "gemv_quantized_int4" : "gemv_quantized_int8",2,nullptr,gws,lws,7,args,info)!=IRIS_SUCCESS ||
        task.flush_out(mr)!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
  }
  auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  printf("Average kernel execution time: %f (us)\n",ns*1e-3/repeat);
  bool ok=true;
  for (unsigned i=0; i<n; ++i) {
    float ref=0;
    for (unsigned j=0; j<n; ++j) {
      int m=int4 ? ((mat[size_t(i)*n/2+j/2]>>((j%2)*4))&15) : static_cast<int8_t>(mat[size_t(i)*n+j]);
      ref += (m-__half2float(zero))*__half2float(scale)*__half2float(vec[j]);
    }
    float actual=__half2float(result[i]);
    if (!std::isfinite(actual) || fabs(ref-actual)>0.125f*n/512) {
      ok=false;
      printf("!!![idx=%u] %f != %f, diff=%f\n",i,actual,ref,ref-actual);
    }
    if (getenv("HECBENCH_DUMP")) printf("NUM int%d %u %.9g\n",int4 ? 4 : 8,i,actual);
  }
  printf("FULL_CHECK int%d rows=%u %s\n",int4 ? 4 : 8,n,ok ? "PASS" : "FAIL");
  // Like the source, reference failures print diagnostics without terminating.
}
void test_gemv_int8_quantized_with_params(unsigned n,unsigned r,unsigned x,unsigned y,float s,float z) {
  puts("GEMV int8 quantized..."); solve(n,r,x,y,s,z,false);
}
void test_gemv_int4_quantized_with_params(unsigned n,unsigned r,unsigned x,unsigned y,float s,float z) {
  puts("GEMV int4 quantized..."); solve(n,r,x,y,s,z,true);
}
