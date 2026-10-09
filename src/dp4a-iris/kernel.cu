#include <stdint.h>

template <typename T>
__device__ __forceinline__ T dot4_scalar(T a, T b, T c)
{
  if constexpr (__is_same(T, unsigned)) {
    const uint8_t* a8 = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* b8 = reinterpret_cast<const uint8_t*>(&b);
    c += a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
  } else {
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    c += a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
  }
  return c;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T sum)
{
  __shared__ T scratch[1024];
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 512; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return scratch[0];
}

template <typename T>
__device__ void dot_body(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ d, size_t n)
{
  T sum = 0;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
    size_t iInOffset = idx * 4;
    for (int i = 0; i < 4; i++) {
      sum = dot4_scalar(a[iInOffset + i], b[iInOffset + i], sum);
    }
  }

  T aggregate = block_reduce_sum(sum);
  if (threadIdx.x == 0) d[blockIdx.x] = aggregate;
}

extern "C" __global__ void dot_product_int(const int* a, const int* b, int* d, size_t n)
{
  dot_body(a, b, d, n);
}

extern "C" __global__ void dot_product2_int(const int* a, const int* b, int* d, size_t n)
{
  dot_body(a, b, d, n);
}

extern "C" __global__ void dot_product_uint(const unsigned* a, const unsigned* b, unsigned* d, size_t n)
{
  dot_body(a, b, d, n);
}

extern "C" __global__ void dot_product2_uint(const unsigned* a, const unsigned* b, unsigned* d, size_t n)
{
  dot_body(a, b, d, n);
}
