#include <hip/hip_runtime.h>

#ifndef PRECISION
#define PRECISION double
#endif

#include "../degrid-cuda/degrid.h"

extern "C" __global__ void degrid_kernel_double(PRECISION2* __restrict__ out,
                                                const PRECISION2* __restrict__ in,
                                                const size_t npts,
                                                const PRECISION2* __restrict__ img_base,
                                                const size_t img_dim,
                                                const PRECISION2* __restrict__ gcf_base)
{
  const int blockIdx_x = blockIdx.x;
  const int blockDim_x = blockDim.x;
  const int threadIdx_x = threadIdx.x;
  const int gridDim_x = gridDim.x;
  const int blockDim_y = blockDim.y;
  const int threadIdx_y = threadIdx.y;
  const PRECISION2* img = img_base + IMG_SIZE * GCF_DIM + GCF_DIM;
  const PRECISION2* gcf = gcf_base + GCF_DIM * (GCF_DIM + 1) / 2;

  for (int n = 32 * blockIdx_x; n < npts; n += 32 * gridDim_x) {
    for (int q = threadIdx_y; q < 32; q += blockDim_y) {
      PRECISION2 inn = in[n + q];
      const int sub_x = floor(GCF_GRID * (inn.x - floor(inn.x)));
      const int sub_y = floor(GCF_GRID * (inn.y - floor(inn.y)));
      const int main_x = floor(inn.x);
      const int main_y = floor(inn.y);
      PRECISION2 sum = { 0, 0 };
      for (int a = threadIdx_x - GCF_DIM / 2; a < GCF_DIM / 2; a += blockDim_x) {
        for (int b = -GCF_DIM / 2; b < GCF_DIM / 2; b++) {
          auto r1 = img[main_x + a + img_dim * (main_y + b)].x;
          auto i1 = img[main_x + a + img_dim * (main_y + b)].y;
          if (main_x + a < 0 || main_y + b < 0 ||
              main_x + a >= img_dim || main_y + b >= img_dim) {
            r1 = i1 = 0;
          }
          auto r2 = gcf[GCF_DIM * GCF_DIM * (GCF_GRID * sub_y + sub_x) +
                        GCF_DIM * b + a].x;
          auto i2 = gcf[GCF_DIM * GCF_DIM * (GCF_GRID * sub_y + sub_x) +
                        GCF_DIM * b + a].y;
          sum.x += r1 * r2 - i1 * i2;
          sum.y += r1 * i2 + r2 * i1;
        }
      }

      for (int s = blockDim_x < 16 ? blockDim_x : 16; s > 0; s /= 2) {
        sum.x += __shfl_down(sum.x, s);
        sum.y += __shfl_down(sum.y, s);
      }
      if (threadIdx_x == 0) {
        out[n + q] = sum;
      }
    }
  }
}
