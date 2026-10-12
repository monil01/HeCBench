#if defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

#ifdef DOUBLE_PRECISION
using T = double;
#else
using T = float;
#endif

extern "C" __global__ void PTTWAC_soa_asta(const int A, const int B, const int b,
                                            T *__restrict__ input,
                                            int *__restrict__ finished,
                                            int *__restrict__ head) {
  __shared__ int lmem[2];
  const int tid = threadIdx.x;
  const int m = A * B - 1;

  if (tid == 0) lmem[1] = atomicAdd(&head[0], 1);
  __syncthreads();

  while (lmem[1] < m) {
    int next_in_cycle = (lmem[1] * A) - m * (lmem[1] / B);
    if (next_in_cycle == lmem[1]) {
      __syncthreads();
      if (tid == 0) lmem[1] = atomicAdd(&head[0], 1);
      __syncthreads();
      continue;
    }

    T data1, data2, data3, data4;
    int i = tid;
    if (i < b) data1 = input[lmem[1] * b + i];
    i += blockDim.x;
    if (i < b) data2 = input[lmem[1] * b + i];
    i += blockDim.x;
    if (i < b) data3 = input[lmem[1] * b + i];
    i += blockDim.x;
    if (i < b) data4 = input[lmem[1] * b + i];

    if (tid == 0) lmem[0] = atomicAdd(&finished[lmem[1]], 0);
    __syncthreads();

    for (; lmem[0] == 0;
         next_in_cycle = (next_in_cycle * A) - m * (next_in_cycle / B)) {
      T backup1, backup2, backup3, backup4;
      i = tid;
      if (i < b) backup1 = input[next_in_cycle * b + i];
      i += blockDim.x;
      if (i < b) backup2 = input[next_in_cycle * b + i];
      i += blockDim.x;
      if (i < b) backup3 = input[next_in_cycle * b + i];
      i += blockDim.x;
      if (i < b) backup4 = input[next_in_cycle * b + i];

      __syncthreads();
      if (tid == 0) lmem[0] = atomicExch(&finished[next_in_cycle], 1);
      __syncthreads();

      if (!lmem[0]) {
        i = tid;
        if (i < b) input[next_in_cycle * b + i] = data1;
        i += blockDim.x;
        if (i < b) input[next_in_cycle * b + i] = data2;
        i += blockDim.x;
        if (i < b) input[next_in_cycle * b + i] = data3;
        i += blockDim.x;
        if (i < b) input[next_in_cycle * b + i] = data4;
      }
      i = tid;
      if (i < b) data1 = backup1;
      i += blockDim.x;
      if (i < b) data2 = backup2;
      i += blockDim.x;
      if (i < b) data3 = backup3;
      i += blockDim.x;
      if (i < b) data4 = backup4;
    }

    if (tid == 0) lmem[1] = atomicAdd(&head[0], 1);
    __syncthreads();
  }
}
