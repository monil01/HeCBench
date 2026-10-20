#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#include <cmath>
template<class T> __global__ void selector_typed(T* a,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]=a[i]*T(2)+T(1);
}
__global__ void calibration_half2(__half2* a,__half2 b,int n,int repeats) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){__half2 x=a[i];
 for(int r=0;r<repeats;r++)x=__hfma2(x,b,b);a[i]=x;}
}
__global__ void calibration_half(__half* a,__half b,int n,int repeats) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){__half x=a[i];
 for(int r=0;r<repeats;r++)x=__hfma(x,b,b);a[i]=x;}
}
__global__ void calibration_tensor(float* out,int repeats) {
 using namespace nvcuda;
 wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
 wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
 wmma::fragment<wmma::accumulator,16,16,16,float> c;
 wmma::fill_fragment(a,__float2half(1.f));wmma::fill_fragment(b,__float2half(1.f));wmma::fill_fragment(c,0.f);
 for(int r=0;r<repeats;r++)wmma::mma_sync(c,a,b,c);
 wmma::store_matrix_sync(out,c,16,wmma::mem_row_major);
}
int main(){
 const int n=4096,R=128;__half2* h2;__half* h;float*f;double*d;float*t;
 cudaMalloc(&h2,n*sizeof(__half2));cudaMalloc(&h,n*sizeof(__half));
 cudaMalloc(&f,n*sizeof(float));cudaMalloc(&d,n*sizeof(double));cudaMalloc(&t,256*sizeof(float));
 cudaMemset(h2,0,n*sizeof(__half2));cudaMemset(h,0,n*sizeof(__half));cudaMemset(f,0,n*sizeof(float));cudaMemset(d,0,n*sizeof(double));
 calibration_half2<<<n/128,128>>>(h2,__float2half2_rn(.5f),n,R);
 calibration_half<<<n/128,128>>>(h,__float2half(.5f),n,R);
 calibration_tensor<<<1,32>>>(t,R);
 for(int r=0;r<3;r++) {selector_typed<<<n/128,128>>>(f,n);selector_typed<<<n/128,128>>>(d,n);
 selector_typed<<<n/256,256>>>(f,n);selector_typed<<<n/256,256>>>(d,n);}
 if(cudaDeviceSynchronize()!=cudaSuccess)return 3;
 std::vector<__half2> v2(n);std::vector<__half> v(n);std::vector<float> vt(256),vf(n);std::vector<double> vd(n);
 cudaMemcpy(v2.data(),h2,n*sizeof(__half2),cudaMemcpyDeviceToHost);cudaMemcpy(v.data(),h,n*sizeof(__half),cudaMemcpyDeviceToHost);
 cudaMemcpy(vt.data(),t,256*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(vf.data(),f,n*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(vd.data(),d,n*sizeof(double),cudaMemcpyDeviceToHost);
 bool ok=true;for(int i=0;i<n;i++)ok &= __low2float(v2[i])==1.f && __high2float(v2[i])==1.f && __half2float(v[i])==1.f && vf[i]==63.f && vd[i]==63.;
 for(float x:vt)ok &= x==2048.f;
 std::printf("calibration_correct=%s N=%d R=%d half2_expected_thread_instructions=%d half2_expected_scalar_flops=%d tensor_expected_scalar_flops=%d\n",ok?"true":"false",n,R,n*R,4*n*R,2*16*16*16*R);
 return ok?0:1;
}
