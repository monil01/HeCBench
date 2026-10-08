#include <hip/hip_runtime.h>

#include <ctime>

#ifndef NUM_THREADS
#define NUM_THREADS 256
#endif

extern "C" __global__ void timedReduction(const float* input, float* output, clock_t* timer) {
  __shared__ float shared[2 * NUM_THREADS];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if (tid == 0) {
    timer[bid] = clock();
  }

  shared[tid] = input[tid];
  shared[tid + blockDim.x] = input[tid + blockDim.x];

  for (int d = blockDim.x; d > 0; d /= 2) {
    __syncthreads();

    if (tid < d) {
      const float f0 = shared[tid];
      const float f1 = shared[tid + d];

      if (f1 < f0) {
        shared[tid] = f1;
      }
    }
  }

  if (tid == 0) {
    output[bid] = shared[0];
  }

  __syncthreads();

  if (tid == 0) {
    timer[bid + gridDim.x] = clock();
  }
}
