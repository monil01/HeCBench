#include <hip/hip_runtime.h>

extern "C" __global__ void add(int n, const float* x, float* y)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] += x[i];
  }
}
