#include <hip/hip_runtime.h>

#include "kernel.h"

extern "C" __global__ void contraction_float(const float* __restrict__ tensor,
                                              const float* __restrict__ adj,
                                              float* __restrict__ value,
                                              const int output_size,
                                              const int N,
                                              const int nChanels)
{
  contraction<float>(tensor, adj, value, output_size, N, nChanels);
}

extern "C" __global__ void contraction_double(const double* __restrict__ tensor,
                                               const double* __restrict__ adj,
                                               double* __restrict__ value,
                                               const int output_size,
                                               const int N,
                                               const int nChanels)
{
  contraction<double>(tensor, adj, value, output_size, N, nChanels);
}
