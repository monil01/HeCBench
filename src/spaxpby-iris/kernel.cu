#include <stddef.h>

extern "C" __global__ void spaxpby_kernel(
    const size_t nnz,
    const float alpha_repeat,
    const size_t* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ y)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nnz) return;
  y[indices[idx]] += alpha_repeat * values[idx];
}
