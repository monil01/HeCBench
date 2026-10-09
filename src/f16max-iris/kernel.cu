#include <cuda.h>
#include <cuda_fp16.h>

__device__ half2 half_max(const half2 a, const half2 b)
{
  const half2 sub = __hsub2(a, b);
  const unsigned sign = (*reinterpret_cast<const unsigned*>(&sub)) & 0x80008000u;
  const unsigned sw = 0x00003210 | (((sign >> 21) | (sign >> 13)) * 0x11);
  const unsigned int res = __byte_perm(*reinterpret_cast<const unsigned*>(&a),
                                       *reinterpret_cast<const unsigned*>(&b), sw);
  return *reinterpret_cast<const half2*>(&res);
}

__device__ half half_max(const half a, const half b)
{
  const half sub = __hsub(a, b);
  const unsigned sign = (*reinterpret_cast<const short*>(&sub)) & 0x8000u;
  const unsigned sw = 0x00000010 | ((sign >> 13) * 0x11);
  const unsigned short res = __byte_perm(*reinterpret_cast<const short*>(&a),
                                         *reinterpret_cast<const short*>(&b), sw);
  return *reinterpret_cast<const half*>(&res);
}

extern "C" __global__ void hmax_half2(
    const half2* __restrict__ a,
    const half2* __restrict__ b,
    half2* __restrict__ r,
    const size_t size)
{
  for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    r[i] = half_max(a[i], b[i]);
  }
}

extern "C" __global__ void hmax_half(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ r,
    const size_t size)
{
  for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    r[i] = half_max(a[i], b[i]);
  }
}
