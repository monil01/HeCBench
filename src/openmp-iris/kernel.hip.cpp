#include <hip/hip_runtime.h>

extern "C" __global__ void addConstant(int* g_a, const int b, const int repeat)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < repeat; i++) {
    g_a[idx] += i % b;
  }
}
