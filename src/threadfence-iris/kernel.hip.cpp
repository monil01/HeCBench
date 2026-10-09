#include <hip/hip_runtime.h>

extern "C" __global__ void sum(
    const float* __restrict__ array,
    const int N,
    unsigned int* __restrict__ count,
    volatile float* __restrict__ result)
{
  __shared__ bool isLastBlockDone;
  __shared__ float partialSum;

  unsigned int bid = blockIdx.x;
  unsigned int num_blocks = gridDim.x;
  unsigned int block_size = blockDim.x;
  unsigned int lid = threadIdx.x;
  unsigned int gid = bid * block_size + lid;

  if (lid == 0) {
    partialSum = 0;
  }
  __syncthreads();

  if (gid < static_cast<unsigned int>(N)) {
    atomicAdd(&partialSum, array[gid]);
  }

  __syncthreads();

  if (lid == 0) {
    result[bid] = partialSum;
    __threadfence();

    unsigned int value = atomicAdd(count, 1);
    isLastBlockDone = (value == (num_blocks - 1));
  }

  __syncthreads();

  if (isLastBlockDone) {
    if (lid == 0) {
      partialSum = 0;
    }
    __syncthreads();

    for (int i = lid; i < static_cast<int>(num_blocks); i += block_size) {
      atomicAdd(&partialSum, result[i]);
    }

    __syncthreads();

    if (lid == 0) {
      result[0] = partialSum;
      *count = 0;
    }
  }
}
