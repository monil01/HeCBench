#include <hip/hip_runtime.h>

extern "C" __global__ void reduceKernel(
    float* d_Result,
    const float* d_Input,
    int N)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadN = gridDim.x * blockDim.x;
  float sum = 0.0f;

  for (int pos = tid; pos < N; pos += threadN) {
    sum += d_Input[pos];
  }

  d_Result[tid] = sum;
}
