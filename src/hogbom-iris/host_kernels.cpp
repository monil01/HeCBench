#include "dmem_runtime.hpp"
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



#include "dispatch.h"
static Peak findPeak(const float* d_image, Peak* d_peak, size_t size)
{
  const int nBlocks = findPeakNBlocks;
  Peak peaks[nBlocks];

  // Find peak
  dispatch_k_findPeak( dim3(nBlocks), dim3(findPeakWidth), d_image, size, d_peak);

  // Get the peaks array back from the device
  iris_copy(&peaks, d_peak, nBlocks * sizeof(Peak), iris_d2h);

  // Each thread block returned a peak, find the absolute maximum
  Peak p;
  p.val = 0.f;
  p.pos = 0;
  for (int i = 0; i < nBlocks; ++i) {
    if (fabsf(peaks[i].val) > fabsf(p.val)) {
      p.val = peaks[i].val;
      p.pos = peaks[i].pos;
    }
  }

  return p;
}



static void subtractPSF(const float* d_psf, const int psfWidth,
    float* d_residual, const int residualWidth,
    const size_t peakPos, const size_t psfPeakPos,
    const float absPeakVal, const float gain)
{
  const int blockDim = 16;

  const int rx = idxToPos(peakPos, residualWidth).x;
  const int ry = idxToPos(peakPos, residualWidth).y;

  const int px = idxToPos(psfPeakPos, psfWidth).x;
  const int py = idxToPos(psfPeakPos, psfWidth).y;

  const int diffx = rx - px;
  const int diffy = ry - px;

  const int startx = std::max(0, rx - px);
  const int starty = std::max(0, ry - py);

  const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
  const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

  // Note: Both start* and stop* locations are inclusive.
  const int blocksx = (stopx-startx + blockDim) / blockDim;
  const int blocksy = (stopy-starty + blockDim) / blockDim;

  dim3 numBlocks(blocksx, blocksy);
  dim3 threadsPerBlock(blockDim, blockDim);
  dispatch_k_subtractPSF( dim3(numBlocks), dim3(threadsPerBlock), d_psf, psfWidth, d_residual, residualWidth,
      startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gain);
}

HogbomTest::HogbomTest()
{
}

HogbomTest::~HogbomTest()
{
}

void HogbomTest::deconvolve(const std::vector<float>& dirty,
    const size_t dirtyWidth,
    const std::vector<float>& psf,
    const size_t psfWidth,
    std::vector<float>& model,
    std::vector<float>& residual)
{
  residual = dirty;

  // Initialise a peaks array on the device. Each thread block will return
  // a peak. Note:  the d_peaks array is not initialized (hence avoiding the
  // memcpy), it is up to the device function to do that
  Peak* d_peaks;
  iris_alloc((void **) &d_peaks, findPeakNBlocks * sizeof(Peak));

  float* d_psf;
  float* d_residual;
  const size_t psf_size = psf.size();
  const size_t residual_size = residual.size();
  iris_alloc((void **) &d_psf, psf_size * sizeof(float));
  iris_alloc((void **) &d_residual, residual_size * sizeof(float));

  iris_copy(d_psf, &psf[0], psf_size * sizeof(float), iris_h2d);
  iris_copy(d_residual, &residual[0], residual_size * sizeof(float), iris_h2d);

  // Find peak of PSF
  Peak psfPeak = findPeak(d_psf, d_peaks, psf_size);

  std::cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val 
    << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
    << idxToPos(psfPeak.pos, psfWidth).y << std::endl;
  assert(psfPeak.pos <= psf_size);

  iris_sync();
  Stopwatch sw;
  sw.start();

  for (unsigned int i = 0; i < niters; ++i) {
    // Find peak in the residual image
    Peak peak = findPeak(d_residual, d_peaks, residual_size);

    assert(peak.pos <= residual_size);

    // Check if threshold has been reached
    if (fabsf(peak.val) < threshold) {
      std::cout << "Reached stopping threshold" << std::endl;
    }

    // Subtract the PSF from the residual image (this function will launch
    // an kernel asynchronously, need to sync later
    subtractPSF(d_psf, psfWidth, d_residual, dirtyWidth, peak.pos, psfPeak.pos, peak.val, gain);

    // Add to model
    model[peak.pos] += peak.val * gain;
  }

  iris_sync();
  const double time = sw.stop();

  // Report on timings
  std::cout << "    Time " << time << " (s) " << std::endl;
  std::cout << "    Time per cycle " << time / niters * 1000 << " (ms)" << std::endl;
  std::cout << "    Cleaning rate  " << niters / time << " (iterations per second)" << std::endl;
  std::cout << "Done" << std::endl;

  // Copy device arrays back into the host 
  iris_copy(&residual[0], d_residual, residual.size() * sizeof(float), iris_d2h);

  // Free device memory
  iris_free(d_peaks);
  iris_free(d_psf);
  iris_free(d_residual);
}
