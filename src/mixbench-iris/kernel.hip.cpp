#include <hip/hip_runtime.h>

#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

extern "C" __global__ void benchmark_func(float* g_data, const int compute_iterations)
{
  const unsigned int blockSize = blockDim.x;
  const int stride = blockSize;
  const int idx = blockIdx.x * blockSize * granularity + threadIdx.x;
  const int big_stride = gridDim.x * blockSize * granularity;

  float tmps[granularity];
  for (int k = 0; k < fusion_degree; k++) {
    #pragma unroll
    for (int j = 0; j < granularity; j++) {
      tmps[j] = g_data[idx + j * stride + k * big_stride];
      for (int i = 0; i < compute_iterations; i++) {
        tmps[j] = tmps[j] * tmps[j] + seed;
      }
    }

    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < granularity; j += 2) {
      sum += tmps[j] * tmps[j + 1];
    }

    #pragma unroll
    for (int j = 0; j < granularity; j++) {
      g_data[idx + k * big_stride] = sum;
    }
  }
}
