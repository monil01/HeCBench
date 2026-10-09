#include <math.h>
#include <stdint.h>

__device__ __inline__ float gelu(float x)
{
  constexpr float kAlpha = M_SQRT1_2;
  return x * 0.5f * (1.0f + erff(x * kAlpha));
}

extern "C" __global__ void geglu_kernel(
    float* out,
    const float* x_and_gate,
    uint64_t total,
    int n,
    int dim_last)
{
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const uint64_t row = idx / static_cast<uint64_t>(dim_last);
  const uint64_t col = idx - row * static_cast<uint64_t>(dim_last);
  const uint64_t x_base = (row * 2) * static_cast<uint64_t>(dim_last) + col;
  const uint64_t gate_base = (row * 2 + 1) * static_cast<uint64_t>(dim_last) + col;
  (void)n;
  out[idx] = x_and_gate[x_base] * gelu(x_and_gate[gate_base]);
}
