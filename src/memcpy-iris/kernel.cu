#include <cuda_runtime.h>

extern "C" __global__ void memcpy_touch_kernel(int* data, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = data[i];
  }
}
