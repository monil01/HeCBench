#include <cuda_runtime.h>

extern "C" __global__ void KernelWithSmallArgs(const char* args, char* out)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (out) *out = args[i];
}

extern "C" __global__ void KernelWithMediumArgs(const char* args, char* out)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (out) *out = args[i];
}

extern "C" __global__ void KernelWithLargeArgs(const char* args, char* out)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (out) *out = args[i];
}
