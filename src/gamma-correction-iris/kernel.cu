#include "utils/ImgPixel.hpp"

extern "C" __global__ void gamma_correction(ImgPixel* pixel)
{
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const float v = (0.3f * pixel[i].r + 0.59f * pixel[i].g + 0.11f * pixel[i].b) / 255.0f;

  std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255.0f * v * v);
  if (gamma_pixel > 255) gamma_pixel = 255;
  pixel[i].set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
}
