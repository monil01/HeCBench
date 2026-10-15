#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
template<class T,class S,class A> __device__ void gemm(const T* a,const T* b,S* c,int m,int n,int k) {
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=size_t(m)*n)return;
 int row=i/n,col=i%n;A sum=0;
 for(int q=0;q<k;++q)sum+=A(a[size_t(row)*k+q])*A(b[size_t(q)*n+col]);
 c[i]=S(sum);
}
#define GEMM(NAME,T,S,A) extern "C" __global__ void NAME(const T* a,const T* b,S* c,int m,int n,int k){gemm<T,S,A>(a,b,c,m,n,k);}
GEMM(gemm_double,double,double,double)
GEMM(gemm_float,float,float,float)
GEMM(gemm_half,__half,__half,float)
GEMM(gemm_bfloat,hip_bfloat16,hip_bfloat16,float)
GEMM(gemm_int,int8_t,int32_t,int32_t)
