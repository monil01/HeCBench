#include <cuda.h>

extern "C" __global__ void kernel_BS(const float* __restrict__ acc_a,
                                      const float* __restrict__ acc_z,
                                      size_t* __restrict__ acc_r,
                                      const size_t zSize,
                                      const size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= zSize) return;
  float z = acc_z[i];
  size_t low = 0;
  size_t high = n;
  while (high - low > 1) {
    size_t mid = low + (high - low) / 2;
    if (z < acc_a[mid])
      high = mid;
    else
      low = mid;
  }
  acc_r[i] = low;
}

extern "C" __global__ void kernel_BS2(const float* __restrict__ acc_a,
                                       const float* __restrict__ acc_z,
                                       size_t* __restrict__ acc_r,
                                       const size_t zSize,
                                       const size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= zSize) return;
  unsigned nbits = 0;
  while (n >> nbits) nbits++;
  size_t k = 1ULL << (nbits - 1);
  float z = acc_z[i];
  size_t idx = (acc_a[k] <= z) ? k : 0;
  while (k >>= 1) {
    size_t r = idx | k;
    if (r < n && z >= acc_a[r]) {
      idx = r;
    }
  }
  acc_r[i] = idx;
}

extern "C" __global__ void kernel_BS3(const float* __restrict__ acc_a,
                                       const float* __restrict__ acc_z,
                                       size_t* __restrict__ acc_r,
                                       const size_t zSize,
                                       const size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= zSize) return;
  unsigned nbits = 0;
  while (n >> nbits) nbits++;
  size_t k = 1ULL << (nbits - 1);
  float z = acc_z[i];
  size_t idx = (acc_a[k] <= z) ? k : 0;
  while (k >>= 1) {
    size_t r = idx | k;
    size_t w = r < n ? r : n;
    if (z >= acc_a[w]) {
      idx = r;
    }
  }
  acc_r[i] = idx;
}

extern "C" __global__ void kernel_BS4(const float* __restrict__ acc_a,
                                       const float* __restrict__ acc_z,
                                       size_t* __restrict__ acc_r,
                                       const size_t zSize,
                                       const size_t n)
{
  __shared__ size_t k;

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= zSize) return;
  size_t lid = threadIdx.x;

  if (lid == 0) {
    unsigned nbits = 0;
    while (n >> nbits) nbits++;
    k = 1ULL << (nbits - 1);
  }
  __syncthreads();

  size_t p = k;
  float z = acc_z[i];
  size_t idx = (acc_a[p] <= z) ? p : 0;
  while (p >>= 1) {
    size_t r = idx | p;
    size_t w = r < n ? r : n;
    if (z >= acc_a[w]) {
      idx = r;
    }
  }
  acc_r[i] = idx;
}
