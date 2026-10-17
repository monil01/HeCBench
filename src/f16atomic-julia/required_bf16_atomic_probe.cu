#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
__global__ void required(__nv_bfloat162* p){__nv_bfloat162 val;*reinterpret_cast<unsigned*>(&val)=0x3f800000U;atomicAdd(p,val);}
int main(){__nv_bfloat162*p;auto status=cudaMalloc(&p,sizeof(*p));if(status!=cudaSuccess)return(int)status;cudaMemset(p,0,sizeof(*p));required<<<1,1>>>(p);status=cudaGetLastError();if(status==cudaSuccess)status=cudaDeviceSynchronize();printf("required_BF16x2_atomic_status=%d %s\n",(int)status,cudaGetErrorString(status));return(int)status;}
