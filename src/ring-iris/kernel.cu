#include <cuda_runtime.h>

extern "C" __global__ void ring_copy(int* dst, const int* src, long len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < static_cast<size_t>(len)) dst[i] = src[i];
}

extern "C" __global__ void ring_init(int* dst, long len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < static_cast<size_t>(len)) dst[i] = static_cast<int>(i);
}
