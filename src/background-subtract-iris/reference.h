#pragma once

#include <cmath>
#include <cstdlib>

inline void merge_ref(
    const size_t imgSize,
    const unsigned char* __restrict__ Img,
    const unsigned char* __restrict__ Img1,
    const unsigned char* __restrict__ Img2,
    unsigned char* __restrict__ Tn,
    unsigned char* __restrict__ Bn)
{
  for (size_t i = 0; i < imgSize; i++) {
    if (std::abs(Img[i] - Img1[i]) <= Tn[i] && std::abs(Img[i] - Img2[i]) <= Tn[i]) {
      Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];
      const float th = 0.92 * Tn[i] + 0.24 * (Img[i] - Bn[i]);
      Tn[i] = std::fmax(th, 20.f);
    }
  }
}
