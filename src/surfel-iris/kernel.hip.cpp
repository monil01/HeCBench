#include <hip/hip_runtime.h>

#define COL_P_X 0
#define COL_P_Y 1
#define COL_P_Z 2
#define COL_N_X 3
#define COL_N_Y 4
#define COL_N_Z 5
#define COL_RSq 6
#define COL_DIM 7

extern "C" __global__ void surfel_render(const float* __restrict__ s, int n, float f,
                                          int w, int h, float* __restrict__ d)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < w && y < h) {
    float rayx = float(x) - (w - 1) * 0.5f;
    float rayy = float(y) - (h - 1) * 0.5f;
    float rayz = f;
    float dMin = 1e20f;
    for (int i = 0; i < n; ++i) {
      float px = s[i * COL_DIM + COL_P_X];
      float py = s[i * COL_DIM + COL_P_Y];
      float pz = s[i * COL_DIM + COL_P_Z];
      float nx = s[i * COL_DIM + COL_N_X];
      float ny = s[i * COL_DIM + COL_N_Y];
      float nz = s[i * COL_DIM + COL_N_Z];
      float rSqMax = s[i * COL_DIM + COL_RSq];
      float pDotn = px * nx + py * ny + pz * nz;
      float dsDotRay = rayx * nx + rayy * ny + rayz * nz;
      float alpha = pDotn / dsDotRay;
      float dx = rayx * alpha - px;
      float dy = rayy * alpha - py;
      float dz = rayz * alpha - pz;
      float t = rayz * alpha;
      float rSq = dx * dx + dy * dy + dz * dz;
      if (rSq < rSqMax && t < dMin) dMin = t;
    }
    d[y * w + x] = dMin > 100.0f ? 0.0f : dMin;
  }
}

extern "C" __global__ void surfel_render_tile(const float* __restrict__ s, int n, float f,
                                               int w, int h, float* __restrict__ d)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  const int tile = 256;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  __shared__ float sh[tile * COL_DIM];

  float rayx = float(x) - (w - 1) * 0.5f;
  float rayy = float(y) - (h - 1) * 0.5f;
  float rayz = f;
  float dMin = 1e20f;

  for (int base = 0; base < n; base += tile) {
    if (tid < tile && base + tid < n) {
      for (int k = 0; k < COL_DIM; ++k) sh[tid * COL_DIM + k] = s[(base + tid) * COL_DIM + k];
    }
    __syncthreads();
    int tileCount = min(tile, n - base);
    for (int i = 0; i < tileCount; ++i) {
      float px = sh[i * COL_DIM + COL_P_X];
      float py = sh[i * COL_DIM + COL_P_Y];
      float pz = sh[i * COL_DIM + COL_P_Z];
      float nx = sh[i * COL_DIM + COL_N_X];
      float ny = sh[i * COL_DIM + COL_N_Y];
      float nz = sh[i * COL_DIM + COL_N_Z];
      float rSqMax = sh[i * COL_DIM + COL_RSq];
      float dsDotRay = rayx * nx + rayy * ny + rayz * nz;
      float pDotn = px * nx + py * ny + pz * nz;
      float alpha = pDotn / dsDotRay;
      float t = rayz * alpha;
      float dx = rayx * alpha - px;
      float dy = rayy * alpha - py;
      float dz = rayz * alpha - pz;
      float rSq = dx * dx + dy * dy + dz * dz;
      if (rSq < rSqMax && t < dMin) dMin = t;
    }
    __syncthreads();
  }
  d[y * w + x] = dMin > 100.0f ? 0.0f : dMin;
}
