#include <stddef.h>

#define THREADS_PER_BLOCK 256

extern "C" __global__ void reduce(const float* inputVec,
                                  double* outputVec,
                                  size_t inputSize,
                                  size_t outputSize)
{
  __shared__ double tmp[THREADS_PER_BLOCK];

  const unsigned int tid = threadIdx.x;
  const size_t globaltid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = globaltid; i < inputSize; i += stride) {
    temp_sum += static_cast<double>(inputVec[i]);
  }
  tmp[tid] = temp_sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      tmp[tid] += tmp[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0 && blockIdx.x < outputSize) {
    outputVec[blockIdx.x] = tmp[0];
  }
}

extern "C" __global__ void reduceFinal(const double* inputVec,
                                       double* result,
                                       size_t inputSize)
{
  __shared__ double tmp[THREADS_PER_BLOCK];

  const unsigned int tid = threadIdx.x;
  double temp_sum = 0.0;
  for (size_t i = tid; i < inputSize; i += blockDim.x) {
    temp_sum += inputVec[i];
  }
  tmp[tid] = temp_sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      tmp[tid] += tmp[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[0] = tmp[0];
  }
}
