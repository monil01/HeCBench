#include <cfloat>
#include <cmath>

__host__ __device__ inline float createRoundingFactor(float max, int n)
{
  float delta = (max * (float)n) / (1.f - 2.f * (float)n * FLT_EPSILON);
  int exp;
  frexpf(delta, &exp);
  return ldexpf(1.f, exp);
}

__host__ __device__ inline float truncateWithRoundingFactor(float roundingFactor, float x)
{
  return (roundingFactor + x) - roundingFactor;
}

extern "C" __global__ void sumArray(
    const float factor,
    const int length,
    const int array_index,
    const float* __restrict__ x,
    float* __restrict__ r,
    const int nArrays)
{
  if (array_index >= nArrays) return;
  const float* arr = x + static_cast<size_t>(array_index) * length;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x) {
    float q = truncateWithRoundingFactor(factor, arr[i]);
    atomicAdd(r + array_index, q);
  }
}

extern "C" __global__ void sumArrays(
    const int nArrays,
    const int length,
    const float* __restrict__ x,
    float* __restrict__ r,
    const float* __restrict__ maxVal)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nArrays; i += blockDim.x * gridDim.x) {
    const float* arr = x + static_cast<size_t>(i) * length;
    float factor = createRoundingFactor(maxVal[i], length);
    float s = 0;
    for (int n = length - 1; n >= 0; n--) {
      s += truncateWithRoundingFactor(factor, arr[n]);
    }
    r[i] = s;
  }
}
