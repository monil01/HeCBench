#include <hip/hip_runtime.h>

extern "C" __global__ void adam_kernel(
    float* __restrict__ p,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ g,
    const float b1,
    const float b2,
    const float eps,
    const float grad_scale,
    const float step_size,
    const int time_step,
    const int vector_size,
    const int mode,
    const float decay)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t totThreads = gridDim.x * blockDim.x;

  for (size_t j = i; j < static_cast<size_t>(vector_size); j += totThreads) {
    for (int t = 1; t <= time_step; t++) {
      float scaled_grad = g[j] / grad_scale;
      m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
      v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;
      float m_corrected = m[j] / (1.f - powf(b1, t));
      float v_corrected = v[j] / (1.f - powf(b2, t));
      float denom;
      if (mode == 0)
        denom = sqrtf(v_corrected + eps);
      else
        denom = sqrtf(v_corrected) + eps;
      float update = (m_corrected / denom) + (decay * p[j]);
      p[j] -= (step_size * update);
    }
  }
}
