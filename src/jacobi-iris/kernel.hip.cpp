#include <hip/hip_runtime.h>

#include "jacobi.h"

extern "C" __global__ void jacobi_step(
    float* __restrict__ f,
    const float* __restrict__ f_old,
    float* __restrict__ error)
{
  __shared__ float f_old_tile[18][18];

  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  f_old_tile[threadIdx.y + 1][threadIdx.x + 1] = f_old[IDX(i, j)];

  if (threadIdx.x == 0 && i >= 1) {
    f_old_tile[threadIdx.y + 1][threadIdx.x + 0] = f_old[IDX(i - 1, j)];
  }
  if (threadIdx.x == 15 && i <= N - 2) {
    f_old_tile[threadIdx.y + 1][threadIdx.x + 2] = f_old[IDX(i + 1, j)];
  }
  if (threadIdx.y == 0 && j >= 1) {
    f_old_tile[threadIdx.y + 0][threadIdx.x + 1] = f_old[IDX(i, j - 1)];
  }
  if (threadIdx.y == 15 && j <= N - 2) {
    f_old_tile[threadIdx.y + 2][threadIdx.x + 1] = f_old[IDX(i, j + 1)];
  }

  __syncthreads();

  float err = 0.0f;
  if (j >= 1 && j <= N - 2 && i >= 1 && i <= N - 2) {
    f[IDX(i, j)] = 0.25f * (f_old_tile[threadIdx.y + 1][threadIdx.x + 2] +
                            f_old_tile[threadIdx.y + 1][threadIdx.x + 0] +
                            f_old_tile[threadIdx.y + 2][threadIdx.x + 1] +
                            f_old_tile[threadIdx.y + 0][threadIdx.x + 1]);
    const float df = f[IDX(i, j)] - f_old_tile[threadIdx.y + 1][threadIdx.x + 1];
    err = df * df;
  }

  for (int offset = 8; offset > 0; offset /= 2) {
    err += __shfl_down(err, offset);
  }

  __shared__ float reduction_array[16];
  if (threadIdx.x == 0) {
    reduction_array[threadIdx.y] = err;
  }

  __syncthreads();

  if (threadIdx.y == 0) {
    err = reduction_array[threadIdx.x];
    for (int offset = 8; offset > 0; offset /= 2) {
      err += __shfl_down(err, offset);
    }
    if (threadIdx.x == 0) {
      atomicAdd(error, err);
    }
  }
}
