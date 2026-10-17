#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
__global__ void required_bf16_pair_multiply(__nv_bfloat162* data){data[0]=__hmul2(data[0],data[0]);}
int main(){cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf("GPU %s sm%d%d\n",p.name,p.major,p.minor);__nv_bfloat162*d;cudaMalloc(&d,sizeof(*d));cudaMemset(d,0,sizeof(*d));required_bf16_pair_multiply<<<1,1>>>(d);cudaError_t launch=cudaGetLastError();cudaError_t sync=cudaDeviceSynchronize();printf("launch=%d %s sync=%d %s\n",(int)launch,cudaGetErrorString(launch),(int)sync,cudaGetErrorString(sync));cudaFree(d);return launch==cudaSuccess&&sync==cudaSuccess?0:1;}
