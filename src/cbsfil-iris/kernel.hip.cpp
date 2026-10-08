#include <hip/hip_runtime.h>

#include <stdint.h>

#define POLE (sqrtf(3.0f) - 2.0f)

typedef unsigned int uint;
typedef unsigned char uchar;

__device__ float initial_causal_coefficient(float* c, uint data_length, int step)
{
  const uint horizon = 12 < data_length ? 12 : data_length;
  float zn = POLE;
  float sum = *c;
  for (uint n = 0; n < horizon; n++) {
    sum += zn * *c;
    zn *= POLE;
    c = reinterpret_cast<float*>(reinterpret_cast<uchar*>(c) + step);
  }
  return sum;
}

__device__ float initial_anti_causal_coefficient(float* c)
{
  return (POLE / (POLE - 1.0f)) * *c;
}

__device__ void convert_to_interpolation_coefficients(float* coeffs,
                                                      uint data_length,
                                                      int step)
{
  const float lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);
  float* c = coeffs;
  float previous_c;

  *c = previous_c = lambda * initial_causal_coefficient(c, data_length, step);
  for (uint n = 1; n < data_length; n++) {
    c = reinterpret_cast<float*>(reinterpret_cast<uchar*>(c) + step);
    *c = previous_c = lambda * *c + POLE * previous_c;
  }

  *c = previous_c = initial_anti_causal_coefficient(c);
  for (int n = static_cast<int>(data_length) - 2; 0 <= n; n--) {
    c = reinterpret_cast<float*>(reinterpret_cast<uchar*>(c) - step);
    *c = previous_c = POLE * (previous_c - *c);
  }
}

extern "C" __global__ void toCoef2DX(float* image, uint pitch, uint width, uint height)
{
  const uint y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y < height) {
    float* line = reinterpret_cast<float*>(reinterpret_cast<uchar*>(image) + y * pitch);
    convert_to_interpolation_coefficients(line, width, sizeof(float));
  }
}

extern "C" __global__ void toCoef2DY(float* image, uint pitch, uint width, uint height)
{
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < width) {
    float* line = image + x;
    convert_to_interpolation_coefficients(line, height, pitch);
  }
}
