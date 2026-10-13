#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;
#include "common.h"
#include "tex2d_compat.h"
extern "C" __global__ void DownscaleKernel(int width, int height, int stride,
                                float *out, const float *texFine_ptr, int texFine_width, int texFine_height, int texFine_stride) {
  Tex2DDesc texFine{texFine_ptr,texFine_width,texFine_height,texFine_stride};

  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;
  if (ix >= width || iy >= height) return;

  float dx = 1.0f / (float)width;
  float dy = 1.0f / (float)height;
  float x  = ((float)ix + 0.5f) * dx;
  float y  = ((float)iy + 0.5f) * dy;

  out[ix + iy * stride] = 0.25f * (tex2D_sw(texFine, x - dx*0.25f, y) +
                                   tex2D_sw(texFine, x + dx*0.25f, y) +
                                   tex2D_sw(texFine, x, y - dy*0.25f) +
                                   tex2D_sw(texFine, x, y + dy*0.25f));
}

extern "C" __global__ void UpscaleKernel(int width, int height, int stride,
                              float scale, float *out, const float *texCoarse_ptr, int texCoarse_width, int texCoarse_height, int texCoarse_stride) {
  Tex2DDesc texCoarse{texCoarse_ptr,texCoarse_width,texCoarse_height,texCoarse_stride};

  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;
  if (ix >= width || iy >= height) return;
  float x = ((float)ix + 0.5f) / (float)width;
  float y = ((float)iy + 0.5f) / (float)height;
  out[ix + iy * stride] = tex2D_sw(texCoarse, x, y) * scale;
}

extern "C" __global__ void WarpingKernel(int width, int height, int stride,
                              const float *u, const float *v,
                              float *out, const float *texToWarp_ptr, int texToWarp_width, int texToWarp_height, int texToWarp_stride) {
  Tex2DDesc texToWarp{texToWarp_ptr,texToWarp_width,texToWarp_height,texToWarp_stride};

  const int ix  = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy  = threadIdx.y + blockIdx.y * blockDim.y;
  const int pos = ix + iy * stride;
  if (ix >= width || iy >= height) return;
  float x = ((float)ix + u[pos] + 0.5f) / (float)width;
  float y = ((float)iy + v[pos] + 0.5f) / (float)height;
  out[pos] = tex2D_sw(texToWarp, x, y);
}

extern "C" __global__ void ComputeDerivativesKernel(int width, int height, int stride,
                                         float *Ix, float *Iy, float *Iz,
                                         const float *texSource_ptr, int texSource_width, int texSource_height, int texSource_stride, const float *texTarget_ptr, int texTarget_width, int texTarget_height, int texTarget_stride) {
  Tex2DDesc texSource{texSource_ptr,texSource_width,texSource_height,texSource_stride};
  Tex2DDesc texTarget{texTarget_ptr,texTarget_width,texTarget_height,texTarget_stride};

  const int ix  = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy  = threadIdx.y + blockIdx.y * blockDim.y;
  const int pos = ix + iy * stride;
  if (ix >= width || iy >= height) return;

  float dx = 1.0f/(float)width, dy = 1.0f/(float)height;
  float x  = ((float)ix + 0.5f) * dx;
  float y  = ((float)iy + 0.5f) * dy;

  float t0, t1;
  t0  = tex2D_sw(texSource, x-2.f*dx, y);
  t0 -= tex2D_sw(texSource, x-1.f*dx, y) * 8.f;
  t0 += tex2D_sw(texSource, x+1.f*dx, y) * 8.f;
  t0 -= tex2D_sw(texSource, x+2.f*dx, y);
  t0 /= 12.f;

  t1  = tex2D_sw(texTarget, x-2.f*dx, y);
  t1 -= tex2D_sw(texTarget, x-1.f*dx, y) * 8.f;
  t1 += tex2D_sw(texTarget, x+1.f*dx, y) * 8.f;
  t1 -= tex2D_sw(texTarget, x+2.f*dx, y);
  t1 /= 12.f;
  Ix[pos] = (t0 + t1) * 0.5f;

  Iz[pos] = tex2D_sw(texTarget, x, y) - tex2D_sw(texSource, x, y);

  t0  = tex2D_sw(texSource, x, y-2.f*dy);
  t0 -= tex2D_sw(texSource, x, y-1.f*dy) * 8.f;
  t0 += tex2D_sw(texSource, x, y+1.f*dy) * 8.f;
  t0 -= tex2D_sw(texSource, x, y+2.f*dy);
  t0 /= 12.f;

  t1  = tex2D_sw(texTarget, x, y-2.f*dy);
  t1 -= tex2D_sw(texTarget, x, y-1.f*dy) * 8.f;
  t1 += tex2D_sw(texTarget, x, y+1.f*dy) * 8.f;
  t1 -= tex2D_sw(texTarget, x, y+2.f*dy);
  t1 /= 12.f;
  Iy[pos] = (t0 + t1) * 0.5f;
}

template <int bx, int by>
__device__ __forceinline__ void JacobiImpl(const float *du0, const float *dv0,
                                const float *Ix, const float *Iy,
                                const float *Iz, int w, int h, int s,
                                float alpha, float *du1, float *dv1) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  volatile __shared__ float du[(bx + 2) * (by + 2)];
  volatile __shared__ float dv[(bx + 2) * (by + 2)];

  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  // position within global memory array
  const int pos = min(ix, w - 1) + min(iy, h - 1) * s;

  // position within shared memory array
  const int shMemPos = threadIdx.x + 1 + (threadIdx.y + 1) * (bx + 2);

  // Load data to shared memory.
  // load tile being processed
  du[shMemPos] = du0[pos];
  dv[shMemPos] = dv0[pos];

  // load necessary neighbouring elements
  // We clamp out-of-range coordinates.
  // It is equivalent to mirroring
  // because we access data only one step away from borders.
  if (threadIdx.y == 0) {
    // beginning of the tile
    const int bsx = blockIdx.x * blockDim.x;
    const int bsy = blockIdx.y * blockDim.y;
    // element position within matrix
    int x, y;
    // element position within linear array
    // gm - global memory
    // sm - shared memory
    int gmPos, smPos;

    x = min(bsx + threadIdx.x, w - 1);
    // row just below the tile
    y = max(bsy - 1, 0);
    gmPos = y * s + x;
    smPos = threadIdx.x + 1;
    du[smPos] = du0[gmPos];
    dv[smPos] = dv0[gmPos];

    // row above the tile
    y = min(bsy + by, h - 1);
    smPos += (by + 1) * (bx + 2);
    gmPos = y * s + x;
    du[smPos] = du0[gmPos];
    dv[smPos] = dv0[gmPos];
  } else if (threadIdx.y == 1) {
    // beginning of the tile
    const int bsx = blockIdx.x * blockDim.x;
    const int bsy = blockIdx.y * blockDim.y;
    // element position within matrix
    int x, y;
    // element position within linear array
    // gm - global memory
    // sm - shared memory
    int gmPos, smPos;

    y = min(bsy + threadIdx.x, h - 1);
    // column to the left
    x = max(bsx - 1, 0);
    smPos = bx + 2 + threadIdx.x * (bx + 2);
    gmPos = x + y * s;

    // check if we are within tile
    if (threadIdx.x < by) {
      du[smPos] = du0[gmPos];
      dv[smPos] = dv0[gmPos];
      // column to the right
      x = min(bsx + bx, w - 1);
      gmPos = y * s + x;
      smPos += bx + 1;
      du[smPos] = du0[gmPos];
      dv[smPos] = dv0[gmPos];
    }
  }

  cg::sync(cta);

  if (ix >= w || iy >= h) return;

  // now all necessary data are loaded to shared memory
  int left, right, up, down;
  left = shMemPos - 1;
  right = shMemPos + 1;
  up = shMemPos + bx + 2;
  down = shMemPos - bx - 2;

  float sumU = (du[left] + du[right] + du[up] + du[down]) * 0.25f;
  float sumV = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25f;

  float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos]) /
               (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

  du1[pos] = sumU - Ix[pos] * frac;
  dv1[pos] = sumV - Iy[pos] * frac;
}
extern "C" __global__ void JacobiIteration(const float *du0, const float *dv0,
 const float *Ix,const float *Iy,const float *Iz,int w,int h,int s,float alpha,float *du1,float *dv1) {
 JacobiImpl<32,6>(du0,dv0,Ix,Iy,Iz,w,h,s,alpha,du1,dv1);
}


extern "C" __global__ void AddKernel(const float *op1, const float *op2, int count,
                          float *sum) {
  const int pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos >= count) return;

  sum[pos] = op1[pos] + op2[pos];
}
