#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" __global__ void base_elementwise_add_kernel(const __half* __restrict__ gA,
                                                        const __half* __restrict__ gB,
                                                        __half* __restrict__ gC,
                                                        const size_t n)
{
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) gC[idx] = gA[idx] + gB[idx];
}

extern "C" __global__ void vectorized_elementwise_add_kernel(const __half* __restrict__ gA,
                                                              const __half* __restrict__ gB,
                                                              __half* __restrict__ gC,
                                                              const size_t n)
{
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) gC[idx] = gA[idx] + gB[idx];
}

extern "C" __global__ void tv_elementwise_add_kernel(const __half* __restrict__ gA,
                                                      const __half* __restrict__ gB,
                                                      __half* __restrict__ gC,
                                                      const int m,
                                                      const int n)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    size_t idx = static_cast<size_t>(row) * n + col;
    gC[idx] = gA[idx] + gB[idx];
  }
}
