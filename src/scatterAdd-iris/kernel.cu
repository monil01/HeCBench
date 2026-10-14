#include <cuda_runtime.h>
#include "original_kernels.hpp"
extern "C" __global__ void scatterAdd_kernel_shared1024(const int* indexes, const float* source, float* outputs, int batch_size, int output_size, int vector_dim) {
  scatterAdd_original<1024>(indexes,source,outputs,batch_size,output_size,vector_dim);
}
extern "C" __global__ void scatterAdd_kernel_shared4096(const int* indexes, const float* source, float* outputs, int batch_size, int output_size, int vector_dim) {
  scatterAdd_original<4096>(indexes,source,outputs,batch_size,output_size,vector_dim);
}
extern "C" __global__ void scatterAdd_kernel_shared16384(const int* indexes, const float* source, float* outputs, int batch_size, int output_size, int vector_dim) {
  scatterAdd_original<16384>(indexes,source,outputs,batch_size,output_size,vector_dim);
}
extern "C" __global__ void scatterAdd2_kernel_shared1024(const int* indexes, const float* source, float* outputs, int batch_size, int output_size, int vector_dim) {
  scatterAdd2_original<1024>(indexes,source,outputs,batch_size,output_size,vector_dim);
}
extern "C" __global__ void scatterAdd2_kernel_shared4096(const int* indexes, const float* source, float* outputs, int batch_size, int output_size, int vector_dim) {
  scatterAdd2_original<4096>(indexes,source,outputs,batch_size,output_size,vector_dim);
}
extern "C" __global__ void scatterAdd2_kernel_shared16384(const int* indexes, const float* source, float* outputs, int batch_size, int output_size, int vector_dim) {
  scatterAdd2_original<16384>(indexes,source,outputs,batch_size,output_size,vector_dim);
}
