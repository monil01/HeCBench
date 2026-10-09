#include <hip/hip_runtime.h>

#include "reference.h"

extern "C" __global__ void matern_kernel(
    const int num_targets,
    const float l,
    const float* __restrict__ sources,
    const float* __restrict__ targets,
    const float* __restrict__ weights,
    float* __restrict__ result)
{
  const int tx = threadIdx.x;
  const int px = blockIdx.x * blockDim.x + tx;
  if (px >= num_targets) return;

  const int ty = threadIdx.y;
  const int py = ty;
  if (py >= SY) return;

  __shared__ float local_result[SX * SY];
  __shared__ float local_targets[SX * 3];
  __shared__ float local_sources[SY * 3];
  __shared__ float local_weights[SY];

  if (ty == 0) {
    for (int i = 0; i < 3; i++) {
      local_targets[tx * 3 + i] = targets[px * 3 + i];
    }
  }

  if (tx == 0) {
    for (int i = 0; i < 3; i++) {
      local_sources[ty * 3 + i] = sources[py * 3 + i];
    }
    local_weights[ty] = weights[ty];
  }

  __syncthreads();

  float squared_diff = 0.f;
  for (int i = 0; i < 3; i++) {
    squared_diff += (local_targets[tx * 3 + i] - local_sources[ty * 3 + i]) *
                    (local_targets[tx * 3 + i] - local_sources[ty * 3 + i]);
  }
  const float diff = sqrtf(squared_diff);

  local_result[tx * SY + ty] =
      (1.f + sqrtf(5.f) * diff / l + 5.f * squared_diff / (3.f * l * l)) *
      expf(-sqrtf(5.f) * diff / l) * local_weights[ty];

  __syncthreads();

  if (ty == 0) {
    float res = 0.f;
    for (int i = 0; i < SY; i++) {
      res += local_result[tx * SY + i];
    }
    result[px] = res;
  }
}
