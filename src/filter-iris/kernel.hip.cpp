#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

extern "C" __global__ void filter(int* __restrict__ dst,
                                   int* __restrict__ nres,
                                   const int* __restrict__ src,
                                   int n)
{
  __shared__ int l_n;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) l_n = 0;
  __syncthreads();

  int d = 0;
  int pos = 0;
  if (i < n) {
    d = src[i];
    if (d > 0) pos = atomicAdd(&l_n, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) l_n = atomicAdd(nres, l_n);
  __syncthreads();

  if (i < n && d > 0) {
    pos += l_n;
    dst[pos] = d;
  }
}

__device__ int atomicAggInc(int* ctr)
{
  auto g = cooperative_groups::coalesced_threads();
  int warp_res = 0;
  if (g.thread_rank() == 0) warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

extern "C" __global__ void filter2(int* __restrict__ dst,
                                    int* __restrict__ nres,
                                    const int* __restrict__ src,
                                    int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n && src[i] > 0) {
    dst[atomicAggInc(nres)] = src[i];
  }
}
