#include <hip/hip_runtime.h>

#define GROUP_SIZE 256

extern "C" __global__ void blockAddition(const float* __restrict__ input,
                                          float* __restrict__ output)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int gid = bid * blockDim.x + tid;
  __shared__ float value;
  if (tid == 0) value = input[bid];
  __syncthreads();
  output[gid] += value;
}

extern "C" __global__ void ScanLargeArrays(float* __restrict__ output,
                                            const float* __restrict__ input,
                                            const unsigned int block_size,
                                            float* __restrict__ sumBuffer)
{
  __shared__ float block[GROUP_SIZE];
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;

  block[2 * tid] = input[2 * gid];
  block[2 * tid + 1] = input[2 * gid + 1];
  __syncthreads();

  float cache0 = block[0];
  float cache1 = cache0 + block[1];
  for (int stride = 1; stride < static_cast<int>(block_size); stride *= 2) {
    if (2 * tid >= stride) {
      cache0 = block[2 * tid - stride] + block[2 * tid];
      cache1 = block[2 * tid + 1 - stride] + block[2 * tid + 1];
    }
    __syncthreads();
    block[2 * tid] = cache0;
    block[2 * tid + 1] = cache1;
    __syncthreads();
  }

  sumBuffer[bid] = block[block_size - 1];
  if (tid == 0) {
    output[2 * gid] = 0;
    output[2 * gid + 1] = block[2 * tid];
  } else {
    output[2 * gid] = block[2 * tid - 1];
    output[2 * gid + 1] = block[2 * tid];
  }
}

extern "C" __global__ void prefixSum(float* __restrict__ output,
                                      const float* __restrict__ input,
                                      const unsigned int block_size)
{
  __shared__ float block[GROUP_SIZE + 1];
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;

  block[2 * tid] = input[2 * gid];
  block[2 * tid + 1] = input[2 * gid + 1];
  __syncthreads();

  float cache0 = block[0];
  float cache1 = cache0 + block[1];
  for (int stride = 1; stride < static_cast<int>(block_size); stride *= 2) {
    if (2 * tid >= stride) {
      cache0 = block[2 * tid - stride] + block[2 * tid];
      cache1 = block[2 * tid + 1 - stride] + block[2 * tid + 1];
    }
    __syncthreads();
    block[2 * tid] = cache0;
    block[2 * tid + 1] = cache1;
    __syncthreads();
  }

  if (tid == 0) {
    output[2 * gid] = 0;
    output[2 * gid + 1] = block[2 * tid];
  } else {
    output[2 * gid] = block[2 * tid - 1];
    output[2 * gid + 1] = block[2 * tid];
  }
}
