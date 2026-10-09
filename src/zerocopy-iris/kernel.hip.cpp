#include <hip/hip_runtime.h>

extern "C" __global__ void vectorAddGPU(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
