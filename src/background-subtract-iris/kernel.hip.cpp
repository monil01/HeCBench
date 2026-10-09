#include <cstdlib>
#include <hip/hip_runtime.h>

extern "C" __global__ void findMovingPixels(
    const size_t imgSize,
    const unsigned char* __restrict__ Img,
    const unsigned char* __restrict__ Img1,
    const unsigned char* __restrict__ Img2,
    const unsigned char* __restrict__ Tn,
    unsigned char* __restrict__ Mp)
{
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= imgSize) return;
  if (abs(Img[i] - Img1[i]) > Tn[i] || abs(Img[i] - Img2[i]) > Tn[i]) {
    Mp[i] = 255;
  } else {
    Mp[i] = 0;
  }
}

extern "C" __global__ void updateBackground(
    const size_t imgSize,
    const unsigned char* __restrict__ Img,
    const unsigned char* __restrict__ Mp,
    unsigned char* __restrict__ Bn)
{
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= imgSize) return;
  if (Mp[i] == 0) Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];
}

extern "C" __global__ void updateThreshold(
    const size_t imgSize,
    const unsigned char* __restrict__ Img,
    const unsigned char* __restrict__ Mp,
    const unsigned char* __restrict__ Bn,
    unsigned char* __restrict__ Tn)
{
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= imgSize) return;
  if (Mp[i] == 0) {
    const float th = 0.92 * Tn[i] + 0.24 * (Img[i] - Bn[i]);
    Tn[i] = fmaxf(th, 20.f);
  }
}

extern "C" __global__ void merge(
    const size_t imgSize,
    const unsigned char* __restrict__ Img,
    const unsigned char* __restrict__ Img1,
    const unsigned char* __restrict__ Img2,
    unsigned char* __restrict__ Tn,
    unsigned char* __restrict__ Bn)
{
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= imgSize) return;
  if (abs(Img[i] - Img1[i]) <= Tn[i] && abs(Img[i] - Img2[i]) <= Tn[i]) {
    Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];
    const float th = 0.92 * Tn[i] + 0.24 * (Img[i] - Bn[i]);
    Tn[i] = fmaxf(th, 20.f);
  }
}
