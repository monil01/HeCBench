#include <hip/hip_runtime.h>
extern "C" __global__ void pingpong_increment(double* data,long n){
 for(long i=blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)data[i]=data[i]+1;
}
