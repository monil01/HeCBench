#include <cuda_runtime.h>
#include <cmath>

#include "distort_common.h"

__device__ inline float getRadialX(float x, float y, const Properties* prop)
{
  x = (x * prop->xscale + prop->xshift);
  y = (y * prop->yscale + prop->yshift);
  return x + ((x - prop->centerX) * prop->K *
    ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
}

__device__ inline float getRadialY(float x, float y, const Properties* prop)
{
  x = (x * prop->xscale + prop->xshift);
  y = (y * prop->yscale + prop->yshift);
  return y + ((y - prop->centerY) * prop->K *
    ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
}

__device__ inline void sampleImageTest(const Pixel* src, float idx0, float idx1,
                                       Pixel& result, const Properties* prop)
{
  if ((idx0 < 0) || (idx1 < 0) || (idx0 > prop->height - 1) || (idx1 > prop->width - 1)) {
    result.x = 0;
    result.y = 0;
    result.z = 0;
    return;
  }

  int idx0_floor = static_cast<int>(floorf(idx0));
  int idx0_ceil = static_cast<int>(ceilf(idx0));
  int idx1_floor = static_cast<int>(floorf(idx1));
  int idx1_ceil = static_cast<int>(ceilf(idx1));

  Pixel s1 = src[(idx0_floor * prop->width) + idx1_floor];
  Pixel s2 = src[(idx0_floor * prop->width) + idx1_ceil];
  Pixel s3 = src[(idx0_ceil * prop->width) + idx1_ceil];
  Pixel s4 = src[(idx0_ceil * prop->width) + idx1_floor];

  float x = idx0 - idx0_floor;
  float y = idx1 - idx1_floor;

  result.x = s1.x * (1.f - x) * (1.f - y) + s2.x * (1.f - x) * y + s3.x * x * y + s4.x * x * (1.f - y);
  result.y = s1.y * (1.f - x) * (1.f - y) + s2.y * (1.f - x) * y + s3.y * x * y + s4.y * x * (1.f - y);
  result.z = s1.z * (1.f - x) * (1.f - y) + s2.z * (1.f - x) * y + s3.z * x * y + s4.z * x * (1.f - y);
}

extern "C" __global__ void barrel_distort(const Pixel* __restrict__ src,
                                           Pixel* __restrict__ dst,
                                           const Properties* __restrict__ prop)
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (w < prop->width && h < prop->height) {
    float x = getRadialX(static_cast<float>(w), static_cast<float>(h), prop);
    float y = getRadialY(static_cast<float>(w), static_cast<float>(h), prop);
    Pixel temp;
    sampleImageTest(src, y, x, temp, prop);
    dst[(h * prop->width) + w] = temp;
  }
}
