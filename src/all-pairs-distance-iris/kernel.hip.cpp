#include <hip/hip_runtime.h>

#define INSTANCES 224
#define ATTRIBUTES 4096
#define THREADS 128

extern "C" __global__ void zero_distance(int* distance, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) distance[i] = 0;
}

__device__ int pair_distance(const char* data, int gx, int gy)
{
  const int tid = threadIdx.x;
  int dist = 0;
  for (int i = tid * 4; i < ATTRIBUTES; i += THREADS * 4) {
    const char4 a = *reinterpret_cast<const char4*>(data + i + ATTRIBUTES * gx);
    const char4 b = *reinterpret_cast<const char4*>(data + i + ATTRIBUTES * gy);
    dist += (a.x != b.x);
    dist += (a.y != b.y);
    dist += (a.z != b.z);
    dist += (a.w != b.w);
  }
  return dist;
}

__device__ void reduce_pair(const char* data, int* distance)
{
  __shared__ int partial[THREADS];
  const int tid = threadIdx.x;
  const int gx = blockIdx.x;
  const int gy = blockIdx.y;
  partial[tid] = pair_distance(data, gx, gy);
  __syncthreads();
  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  if (tid == 0) distance[INSTANCES * gy + gx] = partial[0];
}

extern "C" __global__ void k1(const char* data, int* distance)
{
  reduce_pair(data, distance);
}

extern "C" __global__ void k2(const char* data, int* distance)
{
  reduce_pair(data, distance);
}

extern "C" __global__ void k3(const char* data, int* distance)
{
  reduce_pair(data, distance);
}
