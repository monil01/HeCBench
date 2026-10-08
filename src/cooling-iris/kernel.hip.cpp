#include <hip/hip_runtime.h>

#include "cooling.hpp"

extern "C" __global__ void cool_kernel(const int num,
                                        const Real n,
                                        const Real* __restrict__ T,
                                        Real* __restrict__ r,
                                        const int heat_flag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    r[i] = primordial_cool(n, T[i], heat_flag);
  }
}
