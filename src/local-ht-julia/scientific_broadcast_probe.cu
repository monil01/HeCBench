#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
__device__ int original_bcast_warp(int arg) {
  int laneId = threadIdx.x & warpSize;
  int value;
  if (laneId == 0)      // Note unused variable for
    value = arg;        // all threads except lane 0
  value = __shfl_sync(0xffffffff, value, 0);   // Synchronize all threads in warp, and get "value" from lane 0
  return value;
}
__device__ int fixed_bcast_warp(int arg) {
  int laneId = threadIdx.x & (warpSize - 1);
  int value;
  if (laneId == 0)      // Note unused variable for
    value = arg;        // all threads except lane 0
  value = __shfl_sync(0xffffffff, value, 0);   // Synchronize all threads in warp, and get "value" from lane 0
  return value;
}
__global__ void probe(int*old,int*fixed){int i=blockIdx.x*blockDim.x+threadIdx.x;int value=1000+i;old[i]=original_bcast_warp(value);fixed[i]=fixed_bcast_warp(value);}
int main(){int*a,*b;cudaMalloc(&a,1024*4);cudaMalloc(&b,1024*4);probe<<<2,512>>>(a,b);cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;}std::vector<int>x(1024),y(1024);cudaMemcpy(x.data(),a,4096,cudaMemcpyDeviceToHost);cudaMemcpy(y.data(),b,4096,cudaMemcpyDeviceToHost);printf("{\"original\":[");for(int i=0;i<1024;++i)printf("%s%d",i?",":"",x[i]);printf("],\"fixed\":[");for(int i=0;i<1024;++i)printf("%s%d",i?",":"",y[i]);printf("]}\n");cudaFree(a);cudaFree(b);}
