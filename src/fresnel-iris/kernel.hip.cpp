#include <hip/hip_runtime.h>

#include "fresnel_compat.h"
#include "fresnel_sources.h"

extern "C" __global__ void fresnel_kernel(const double* input, double* output, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = Fresnel_Sine_Integral(input[i]);
  }
}
