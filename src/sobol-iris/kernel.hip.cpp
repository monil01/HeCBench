#include <hip/hip_runtime.h>

#include "sobol.h"

#define k_2powneg32 2.3283064E-10F

extern "C" __global__ void sobolGPU_kernel(unsigned n_vectors,
                                           unsigned n_dimensions,
                                           const unsigned* __restrict__ d_directions,
                                           float* __restrict__ d_output)
{
  __shared__ unsigned int v[n_directions];

  d_directions += n_directions * blockIdx.y;
  d_output += n_vectors * blockIdx.y;

  if (threadIdx.x < n_directions) {
    v[threadIdx.x] = d_directions[threadIdx.x];
  }

  __syncthreads();

  int i0 = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  unsigned int g = i0 ^ (i0 >> 1);
  unsigned int X = 0;
  unsigned int mask;

  for (unsigned int k = 0; k < static_cast<unsigned int>(__ffs(stride) - 1); ++k) {
    mask = -(g & 1);
    X ^= mask & v[k];
    g >>= 1;
  }

  if (i0 < static_cast<int>(n_vectors)) {
    d_output[i0] = static_cast<float>(X) * k_2powneg32;
  }

  unsigned int v_log2stridem1 = v[__ffs(stride) - 2];
  unsigned int v_stridemask = stride - 1;

  for (unsigned int i = i0 + stride; i < n_vectors; i += stride) {
    X ^= v_log2stridem1 ^ v[__ffs(~((i - stride) | v_stridemask)) - 1];
    d_output[i] = static_cast<float>(X) * k_2powneg32;
  }
}
