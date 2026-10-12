#include <cuda.h>

#define BLOCK_X 32
#define BLOCK_Y 8
#define IOFF 1
#define JOFF (BLOCK_X + 2)
#define KOFF (BLOCK_X + 2) * (BLOCK_Y + 2)
#define INDEX(i, j, j_off) (i + (j) * (j_off))

extern "C" __global__ void laplace3d(int nx,
                                      int ny,
                                      int nz,
                                      int pitch,
                                      const float* __restrict__ d_u1,
                                      float* __restrict__ d_u2)
{
  int indg, indg_h, indg0;
  int i, j, k, ind, ind_h, halo, active;
  float u2;
  const float sixth = 1.0f / 6.0f;

  const int nxm1 = nx - 1;
  const int nym1 = ny - 1;
  const int nzm1 = nz - 1;
  __shared__ float u1[3 * KOFF];

  k = threadIdx.x + threadIdx.y * BLOCK_X;
  halo = k < 2 * (BLOCK_X + BLOCK_Y + 2);
  if (halo) {
    if (threadIdx.y < 2) {
      i = threadIdx.x;
      j = threadIdx.y * (BLOCK_Y + 1) - 1;
    } else {
      i = (k % 2) * (BLOCK_X + 1) - 1;
      j = k / 2 - BLOCK_X - 1;
    }
    ind_h = INDEX(i + 1, j + 1, JOFF) + KOFF;
    i = INDEX(i, blockIdx.x, BLOCK_X);
    j = INDEX(j, blockIdx.y, BLOCK_Y);
    indg_h = INDEX(i, j, pitch);
    halo = (i >= 0) && (i < nx) && (j >= 0) && (j < ny);
  }

  i = threadIdx.x;
  j = threadIdx.y;
  ind = INDEX(i + 1, j + 1, JOFF) + KOFF;
  i = INDEX(i, blockIdx.x, BLOCK_X);
  j = INDEX(j, blockIdx.y, BLOCK_Y);
  indg = INDEX(i, j, pitch);
  active = (i < nx) && (j < ny);

  if (active) u1[ind + KOFF] = d_u1[indg];
  if (halo) u1[ind_h + KOFF] = d_u1[indg_h];

  for (k = 0; k < nz; k++) {
    if (active) {
      indg0 = indg;
      indg = INDEX(indg, ny, pitch);
      u1[ind - KOFF] = u1[ind];
      u1[ind] = u1[ind + KOFF];
      if (k < nzm1) u1[ind + KOFF] = d_u1[indg];
    }
    if (halo) {
      indg_h = INDEX(indg_h, ny, pitch);
      u1[ind_h - KOFF] = u1[ind_h];
      u1[ind_h] = u1[ind_h + KOFF];
      if (k < nzm1) u1[ind_h + KOFF] = d_u1[indg_h];
    }
    __syncthreads();

    if (active) {
      if (i == 0 || i == nxm1 || j == 0 || j == nym1 || k == 0 || k == nzm1) {
        u2 = u1[ind];
      } else {
        u2 = (u1[ind - IOFF] + u1[ind + IOFF] +
              u1[ind - JOFF] + u1[ind + JOFF] +
              u1[ind - KOFF] + u1[ind + KOFF]) * sixth;
      }
      d_u2[indg0] = u2;
    }
    __syncthreads();
  }
}
