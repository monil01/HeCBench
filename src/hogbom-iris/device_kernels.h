#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstddef>
#include "hogbom.h"
#include "timer.h"

// grids and blocks are constant for the findPeak kernel
#define findPeakNBlocks 128
#define findPeakWidth 256

struct Peak {
  size_t pos;
  float val;
};

struct Position {
  __host__ __device__
    Position(int _x, int _y) : x(_x), y(_y) { };
  int x;
  int y;
};

__host__ __device__
static Position idxToPos(const size_t idx, const int width)
{
  const int y = idx / width;
  const int x = idx % width;
  return Position(x, y);
}

__device__
static size_t posToIdx(const int width, const Position& pos)
{
  return (pos.y * width) + pos.x;
}

extern "C" __global__
void k_findPeak(
  const float *__restrict__ image, 
  const size_t size,
  Peak *__restrict__ absPeak)
{
  __shared__ float maxVal[findPeakWidth];
  __shared__ size_t maxPos[findPeakWidth];
  float val = 0.f;
  size_t pos = 0;

  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
              idx < size; idx += findPeakWidth * findPeakNBlocks) {
    if (fabsf(image[idx]) > fabsf(val)) {
      val = image[idx];
      pos = idx;
    }
  }
  maxVal[threadIdx.x] = val;
  maxPos[threadIdx.x] = pos;

  __syncthreads();

  if (threadIdx.x == 0) {
    val = 0.f, pos = 0;
    for (int i = 0; i < findPeakWidth; ++i) {
      if (fabsf(maxVal[i]) > fabsf(val)) {
        val = maxVal[i];
        pos = maxPos[i];
      }
    }
    absPeak[blockIdx.x].val = val;
    absPeak[blockIdx.x].pos = pos;
  }
}
extern "C" __global__
void k_subtractPSF(
    const float *__restrict__ d_psf,
    const int psfWidth,
          float *__restrict__ d_residual,
    const int residualWidth,
    const int startx, const int starty,
    const int stopx, const int stopy,
    const int diffx, const int diffy,
    const float absPeakVal, const float gain)
{   
  const int x = startx + threadIdx.x + blockIdx.x * blockDim.x;
  const int y = starty + threadIdx.y + blockIdx.y * blockDim.y;

  // thread blocks are of size 16, but the workload is not always a multiple of 16
  if (x <= stopx && y <= stopy) {
    d_residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
      * d_psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
  }
}
