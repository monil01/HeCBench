#include <hip/hip_runtime.h>
#include <cmath>

#define a1 1.5f
#define a2 2.f
#define a3 2.5f
#define a4 3.f
#define a5 3.5f
#define R_min 0.f
#define R_max 2.f
#define BLOCK_SIZE 256

extern "C" __global__ void AIDW_Kernel(const float* __restrict dx,
                                        const float* __restrict dy,
                                        const float* __restrict dz,
                                        const int dnum,
                                        const float* __restrict ix,
                                        const float* __restrict iy,
                                        float* __restrict iz,
                                        const int inum,
                                        const float area,
                                        const float* __restrict avg_dist) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= inum) return;

  float sum = 0.f, dist = 0.f, t = 0.f, z = 0.f, alpha = 1.f;
  float r_obs = avg_dist[tid];
  float r_exp = 0.5f / sqrtf(dnum / area);
  float R_S0 = r_obs / r_exp;

  float u_R = 0.f;
  if (R_S0 >= R_min) u_R = 0.5f - 0.5f * cosf(3.1415926f / R_max * (R_S0 - R_min));
  if (R_S0 >= R_max) u_R = 1.f;

  if (u_R >= 0.f && u_R <= 0.1f) alpha = a1;
  if (u_R > 0.1f && u_R <= 0.3f) alpha = a1 * (1.f - 5.f * (u_R - 0.1f)) + a2 * 5.f * (u_R - 0.1f);
  if (u_R > 0.3f && u_R <= 0.5f) alpha = a3 * 5.f * (u_R - 0.3f) + a1 * (1.f - 5.f * (u_R - 0.3f));
  if (u_R > 0.5f && u_R <= 0.7f) alpha = a3 * (1.f - 5.f * (u_R - 0.5f)) + a4 * 5.f * (u_R - 0.5f);
  if (u_R > 0.7f && u_R <= 0.9f) alpha = a5 * 5.f * (u_R - 0.7f) + a4 * (1.f - 5.f * (u_R - 0.7f));
  if (u_R > 0.9f && u_R <= 1.f) alpha = a5;
  alpha *= 0.5f;

  for (int j = 0; j < dnum; j++) {
    dist = (ix[tid] - dx[j]) * (ix[tid] - dx[j]) + (iy[tid] - dy[j]) * (iy[tid] - dy[j]);
    t = 1.f / powf(dist, alpha);
    sum += t;
    z += dz[j] * t;
  }
  iz[tid] = z / sum;
}

extern "C" __global__ void AIDW_Kernel_Tiled(const float* __restrict dx,
                                              const float* __restrict dy,
                                              const float* __restrict dz,
                                              const int dnum,
                                              const float* __restrict ix,
                                              const float* __restrict iy,
                                              float* __restrict iz,
                                              const int inum,
                                              const float area,
                                              const float* __restrict avg_dist) {
  __shared__ float sdx[BLOCK_SIZE];
  __shared__ float sdy[BLOCK_SIZE];
  __shared__ float sdz[BLOCK_SIZE];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= inum) return;

  float dist = 0.f, t = 0.f, alpha = 0.f;
  int part = (dnum - 1) / BLOCK_SIZE;
  float sum_up = 0.f;
  float sum_dn = 0.f;

  float r_obs = avg_dist[tid];
  float r_exp = 0.5f / sqrtf(dnum / area);
  float R_S0 = r_obs / r_exp;

  float u_R = 0.f;
  if (R_S0 >= R_min) u_R = 0.5f - 0.5f * cosf(3.1415926f / R_max * (R_S0 - R_min));
  if (R_S0 >= R_max) u_R = 1.f;

  if (u_R >= 0.f && u_R <= 0.1f) alpha = a1;
  if (u_R > 0.1f && u_R <= 0.3f) alpha = a1 * (1.f - 5.f * (u_R - 0.1f)) + a2 * 5.f * (u_R - 0.1f);
  if (u_R > 0.3f && u_R <= 0.5f) alpha = a3 * 5.f * (u_R - 0.3f) + a1 * (1.f - 5.f * (u_R - 0.3f));
  if (u_R > 0.5f && u_R <= 0.7f) alpha = a3 * (1.f - 5.f * (u_R - 0.5f)) + a4 * 5.f * (u_R - 0.5f);
  if (u_R > 0.7f && u_R <= 0.9f) alpha = a5 * 5.f * (u_R - 0.7f) + a4 * (1.f - 5.f * (u_R - 0.7f));
  if (u_R > 0.9f && u_R <= 1.f) alpha = a5;
  alpha *= 0.5f;

  float six_t = ix[tid];
  float siy_t = iy[tid];
  int lid = threadIdx.x;
  for (int m = 0; m <= part; m++) {
    int num_threads = min(BLOCK_SIZE, dnum - BLOCK_SIZE * m);
    if (lid < num_threads) {
      sdx[lid] = dx[lid + BLOCK_SIZE * m];
      sdy[lid] = dy[lid + BLOCK_SIZE * m];
      sdz[lid] = dz[lid + BLOCK_SIZE * m];
    }
    __syncthreads();

    for (int e = 0; e < BLOCK_SIZE; e++) {
      float six_s = six_t - sdx[e];
      float siy_s = siy_t - sdy[e];
      dist = six_s * six_s + siy_s * siy_s;
      t = 1.f / powf(dist, alpha);
      sum_dn += t;
      sum_up += t * sdz[e];
    }
    __syncthreads();
  }
  iz[tid] = sum_up / sum_dn;
}
