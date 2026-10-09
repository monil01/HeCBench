#include <hip/hip_runtime.h>

constexpr int m_size = 512 * 8;
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int K = m_size / 2;

extern "C" __global__ void minkowski(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    const float p,
    const float one_over_p)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < K && row < M) {
    float sum = 0;
    for (int i = 0; i < N; i++) {
      sum += powf(fabsf(a[row * N + i] - b[i * K + col]), p);
    }
    c[row * K + col] = powf(sum, one_over_p);
  }
}
