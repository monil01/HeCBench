#include <cuda_runtime.h>

#define RADIUS 7
#define BLOCK_SIZE 256

extern "C" __global__ void stencil_1d(const int* __restrict__ in, int* __restrict__ out)
{
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  temp[lindex] = in[gindex];

  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  __syncthreads();

  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++) {
    result += temp[lindex + offset];
  }

  out[gindex] = result;
}
