#include <hip/hip_runtime.h>

#include <math.h>

extern "C" __global__ void entropy(
    float* __restrict__ d_entropy,
    const char* __restrict__ d_val,
    int height,
    int width)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  char count[16];
  for (int i = 0; i < 16; i++) count[i] = 0;

  char total = 0;

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      int xx = x + dx;
      int yy = y + dy;
      if (xx >= 0 && yy >= 0 && yy < height && xx < width) {
        count[d_val[yy * width + xx]]++;
        total++;
      }
    }
  }

  float e = 0.0f;
  if (total < 1) {
    total = 1;
  } else {
    for (int k = 0; k < 16; k++) {
      float p = __fdividef(static_cast<float>(count[k]), static_cast<float>(total));
      e -= p * log2f(p);
    }
  }

  if (y < height && x < width) d_entropy[y * width + x] = e;
}

extern "C" __global__ void entropy_opt_16_16(
    float* __restrict__ d_entropy,
    const char* __restrict__ d_val,
    const float* __restrict__ d_logTable,
    int height,
    int width)
{
  __shared__ int sd_count[16][16 * 16];

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int idx = threadIdx.y * 16 + threadIdx.x;

  for (int i = 0; i < 16; i++) sd_count[i][idx] = 0;

  char total = 0;
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      int xx = x + dx;
      int yy = y + dy;

      if (xx >= 0 && yy >= 0 && yy < height && xx < width) {
        sd_count[d_val[yy * width + xx]][idx]++;
        total++;
      }
    }
  }

  float e = 0.0f;
  for (int k = 0; k < 16; k++) {
    e -= d_logTable[sd_count[k][idx]];
  }

  e = e / total + log2f(total);
  if (y < height && x < width) d_entropy[y * width + x] = e;
}
