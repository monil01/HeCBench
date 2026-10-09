#include <hip/hip_runtime.h>

#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float michalewicz(const float* xValues, const int dim)
{
  float result = 0.0f;
  for (int i = 0; i < dim; ++i) {
    float a = sinf(xValues[i]);
    float b = sinf(((i + 1) * xValues[i] * xValues[i]) / static_cast<float>(M_PI));
    float c = powf(b, 20);
    result += a * c;
  }
  return -1.0f * result;
}

extern "C" __global__ void eval(const float* values,
                                 float* minima,
                                 const size_t nVectors,
                                 const int dim)
{
  size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < nVectors) {
    atomicMin(minima, michalewicz(values + n * dim, dim));
  }
}
