#include <hip/hip_runtime.h>
#include <cmath>
template<class T> __device__ int signum(T x) { return (T(0)<x)-(x<T(0)); }
template<class T> __device__ void forward(const T* x,const T* y,T* out,T* sd,T* ss,int B,int M,int N,int D) {
  size_t q=blockIdx.x*blockDim.x+threadIdx.x;
  if(q>=size_t(B)*M*N) return;
  size_t b=q/(M*N),i=q/N%M,j=q%N;
  T a=0,s=0;
  for(int k=0;k<D;++k) { a+=fabs(x[(b*M+i)*D+k]-y[(b*N+j)*D+k]); s+=fabs(x[(b*M+i)*D+k]+y[(b*N+j)*D+k]); }
  out[q]=a/s; sd[q]=a; ss[q]=s;
}
template<class T> __device__ void backward(const T* x,const T* y,const T* go,const T* sd,const T* ss,T* gx,int B,int M,int N,int D,int which) {
  size_t q=blockIdx.x*blockDim.x+threadIdx.x;
  int rows=which?N:M, others=which?M:N;
  if(q>=size_t(B)*rows*D) return;
  size_t b=q/(rows*D),r=q/D%rows,k=q%D;
  T val=0;
  for(int z=0;z<others;++z) {
    size_t i=which?z:r,j=which?r:z,p=(b*M+i)*N+j;
    T a=x[(b*M+i)*D+k],c=y[(b*N+j)*D+k];
    val+=go[p]*((which?-1:1)*signum(a-c)/ss[p]-signum(a+c)*sd[p]/ss[p]/ss[p]);
  }
  gx[q]=val;
}
#define KERNELS(S,T) \
extern "C" __global__ void forward_##S(const T*x,const T*y,T*o,T*sd,T*ss,int B,int M,int N,int D){forward(x,y,o,sd,ss,B,M,N,D);} \
extern "C" __global__ void backward_##S(const T*x,const T*y,const T*g,const T*sd,const T*ss,T*gx,int B,int M,int N,int D,int w){backward(x,y,g,sd,ss,gx,B,M,N,D,w);}
KERNELS(f,float)
KERNELS(d,double)
