#include <hip/hip_runtime.h>

#define ATAN2_HD __device__ __host__
#include "atan2_common.h"

extern "C" __global__ void compute_f(const int n,
                                      const float *x,
                                      const float *y,
                                      float *r)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float vy = y[i];
  const float vx = x[i];
  r[i] = safe_atan2f< 3>(vy, vx) +
         safe_atan2f< 5>(vy, vx) +
         safe_atan2f< 7>(vy, vx) +
         safe_atan2f< 9>(vy, vx) +
         safe_atan2f<11>(vy, vx) +
         safe_atan2f<13>(vy, vx) +
         safe_atan2f<15>(vy, vx);
}

extern "C" __global__ void compute_s(const int n,
                                      const float *x,
                                      const float *y,
                                      short *r)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float vy = y[i];
  const float vx = x[i];
  r[i] = unsafe_atan2s< 3>(vy, vx) +
         unsafe_atan2s< 5>(vy, vx) +
         unsafe_atan2s< 7>(vy, vx) +
         unsafe_atan2s< 9>(vy, vx);
}

extern "C" __global__ void compute_i(const int n,
                                      const float *x,
                                      const float *y,
                                      int *r)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float vy = y[i];
  const float vx = x[i];
  r[i] = unsafe_atan2i< 3>(vy, vx) +
         unsafe_atan2i< 5>(vy, vx) +
         unsafe_atan2i< 7>(vy, vx) +
         unsafe_atan2i< 9>(vy, vx) +
         unsafe_atan2i<11>(vy, vx) +
         unsafe_atan2i<13>(vy, vx) +
         unsafe_atan2i<15>(vy, vx);
}
