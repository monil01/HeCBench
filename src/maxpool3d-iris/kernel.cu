#include <cuda_runtime.h>

using DTYPE = float;

extern "C" __global__ void maxpool3d(const DTYPE* i_img, DTYPE* o_img,
                                      const int Hstride, const int Vstride,
                                      const int pool_width, const int pool_height,
                                      const int i_img_count, const int i_img_width,
                                      const int i_img_height, const int o_img_width)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const int o_img_height = i_img_height / Vstride;
  if (x >= o_img_width || y >= o_img_height || z >= i_img_count) return;

  const int xidx = Hstride * x;
  const int yidx = Vstride * y;
  DTYPE maxval = static_cast<DTYPE>(0);
  for (int r = 0; r < pool_height; r++) {
    const int idxIntmp = ((z * i_img_height + yidx + r) * i_img_width) + xidx;
    for (int c = 0; c < pool_width; c++) {
      maxval = fmaxf(maxval, i_img[idxIntmp + c]);
    }
  }
  o_img[(((z * o_img_height) + y) * o_img_width) + x] = maxval;
}
