#include <math.h>

#define BLOCK_SIZE 256

extern "C" __global__ void norm2_partial(double* partial, const float* a, int n)
{
  __shared__ double cache[BLOCK_SIZE];
  const int tid = static_cast<int>(threadIdx.x);
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  double value = 0.0;
  if (idx < n) {
    const double x = static_cast<double>(a[idx]);
    value = x * x;
  }
  cache[tid] = value;
  __syncthreads();
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) cache[tid] += cache[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = cache[0];
}

extern "C" __global__ void norm2_final(float* result, const double* partial, int num_blocks)
{
  __shared__ double cache[BLOCK_SIZE];
  const int tid = static_cast<int>(threadIdx.x);
  double sum = 0.0;
  for (int i = tid; i < num_blocks; i += BLOCK_SIZE) sum += partial[i];
  cache[tid] = sum;
  __syncthreads();
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) cache[tid] += cache[tid + stride];
    __syncthreads();
  }
  if (tid == 0) *result = static_cast<float>(sqrt(cache[0]));
}
