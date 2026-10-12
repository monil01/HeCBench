#include <hip/hip_runtime.h>

#include <cstddef>

extern "C" __global__ void rotary_kernel(float* __restrict__ o1,
                                         float* __restrict__ o2,
                                         const float* __restrict__ x1,
                                         const float* __restrict__ x2,
                                         const float* __restrict__ cosv,
                                         const float* __restrict__ sinv,
                                         size_t n)
{
  constexpr int thread_work_size = 4;
  const size_t block_base = static_cast<size_t>(blockIdx.x) * blockDim.x * thread_work_size;
  for (int j = 0; j < thread_work_size; j++) {
    const size_t i = block_base + threadIdx.x + static_cast<size_t>(j) * blockDim.x;
    if (i < n) {
      o1[i] = x1[i] * cosv[i] - x2[i] * sinv[i];
      o2[i] = x1[i] * sinv[i] + x2[i] * cosv[i];
    }
  }
}
