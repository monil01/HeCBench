#include <hip/hip_runtime.h>
extern "C" __global__ void kernel_add_segment(const float* src,float* dst,size_t offset,size_t length){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<length;i+=size_t(blockDim.x)*gridDim.x)dst[offset+i]+=src[i];
}
extern "C" __global__ void copy_data(const float* src,float* dst,size_t n){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)dst[i]=src[i];
}
