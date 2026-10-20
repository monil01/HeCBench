#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdio>
__global__ void calibration_hadd2(__half2*a,__half2 b,int n,int R){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){__half2 x=a[i];
 for(int r=0;r<R;r++)x=__hadd2(x,b);a[i]=x;}}
__global__ void calibration_hmul2(__half2*a,__half2 b,int n,int R){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){__half2 x=a[i];
 for(int r=0;r<R;r++)x=__hmul2(x,b);a[i]=x;}}
int main(){const int n=4096,R=128;__half2*a,*b;cudaMalloc(&a,n*4);cudaMalloc(&b,n*4);
 std::vector<__half2> h(n,__float2half2_rn(1.f)),got(n);cudaMemset(a,0,n*4);cudaMemcpy(b,h.data(),n*4,cudaMemcpyHostToDevice);
 calibration_hadd2<<<n/128,128>>>(a,__float2half2_rn(1.f/64),n,R);
 calibration_hmul2<<<n/128,128>>>(b,__float2half2_rn(.5f),n,R);
 if(cudaDeviceSynchronize()!=cudaSuccess)return 3;bool ok=true;
 cudaMemcpy(got.data(),a,n*4,cudaMemcpyDeviceToHost);for(auto x:got)ok &= __low2float(x)==2.f&&__high2float(x)==2.f;
 cudaMemcpy(got.data(),b,n*4,cudaMemcpyDeviceToHost);for(auto x:got)ok &= __low2float(x)==0.f&&__high2float(x)==0.f;
 printf("calibration_correct=%s N=%d R=%d expected_each_thread_instructions=%d expected_each_packed_scalar_flops=%d\n",ok?"true":"false",n,R,n*R,2*n*R);return ok?0:1;}
