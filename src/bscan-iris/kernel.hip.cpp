#include <hip/hip_runtime.h>

__device__ __inline__ int warp_scan(int val, volatile int* s_data)
{
  int idx = 2 * threadIdx.x - (threadIdx.x & (warpSize - 1));
  s_data[idx] = 0;
  idx += warpSize;
  int t = s_data[idx] = val;
  s_data[idx] = t += s_data[idx - 1];
  s_data[idx] = t += s_data[idx - 2];
  s_data[idx] = t += s_data[idx - 4];
  s_data[idx] = t += s_data[idx - 8];
  s_data[idx] = t += s_data[idx - 16];
  if (warpSize > 32) s_data[idx] = t += s_data[idx - 32];
  return s_data[idx - 1];
}

__device__ __inline__ unsigned long long lanemask_lt()
{
  const unsigned int lane = threadIdx.x & (warpSize - 1);
  return (1ULL << lane) - 1;
}

__device__ __inline__ unsigned int binary_warp_scan(bool p)
{
  const unsigned long long mask = lanemask_lt();
  unsigned long long b = __ballot(p);
  return __popcll(b & mask);
}

__device__ __inline__ bool valid(int x)
{
  return x > 0;
}

__device__ __inline__ int block_binary_prefix_sums(int x)
{
  __shared__ int sdata[128];

  bool predicate = valid(x);
  int warpPrefix = binary_warp_scan(predicate);
  int idx = threadIdx.x;
  int warpIdx = idx / warpSize;
  int laneIdx = idx & (warpSize - 1);

  if (laneIdx == warpSize - 1) sdata[warpIdx] = warpPrefix + predicate;
  __syncthreads();

  if (idx < warpSize) sdata[idx] = warp_scan(sdata[idx], sdata);
  __syncthreads();

  return warpPrefix + sdata[warpIdx];
}

extern "C" __global__ void binary_scan(int* __restrict__ g_odata,
                                        const int* __restrict__ g_idata)
{
  int i = threadIdx.x;
  g_odata[i] = block_binary_prefix_sums(g_idata[i]);
}
