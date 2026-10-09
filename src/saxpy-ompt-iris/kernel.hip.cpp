#include <hip/hip_runtime.h>

extern "C" __global__ void saxpy_k(int n, float a, const float* __restrict__ x,
                                   float* __restrict__ y)
{
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) y[i] = a * x[i] + y[i];
}
