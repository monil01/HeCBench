#include <cuda.h>
#include <cuda_fp8.h>

#include <cstdint>

__device__ inline float fp32_from_bits(uint32_t w) {
  return __uint_as_float(static_cast<unsigned int>(w));
}

__device__ inline uint32_t fp32_to_bits(float f) {
  return static_cast<uint32_t>(__float_as_uint(f));
}

inline __device__ uint8_t fp8e5m2_from_fp32_value(float f) {
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else if (f_bits < (UINT32_C(113) << 23)) {
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    uint32_t mant_odd = (f_bits >> 21) & 1;
    f_bits += (static_cast<uint32_t>(15 - 127) << 23) + 0xFFFFF;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 21);
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

inline __device__ uint8_t fp8e4m3fn_from_fp32_value(float f) {
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    result = 0x7f;
  } else if (f_bits < (UINT32_C(121) << 23)) {
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    uint8_t mant_odd = (f_bits >> 20) & 1;
    f_bits += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFF;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

extern "C" __global__ void ref_fp32_cvt_e5m2fnuz_kernel(
    uint8_t* __restrict__ dst, const float* __restrict__ src, const int nelems) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) dst[i] = fp8e5m2_from_fp32_value(src[i]);
}

extern "C" __global__ void ref_fp32_cvt_e4m3fnuz_kernel(
    uint8_t* __restrict__ dst, const float* __restrict__ src, const int nelems) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) dst[i] = fp8e4m3fn_from_fp32_value(src[i]);
}

extern "C" __global__ void fp32_cvt_e4m3_kernel(uint8_t* __restrict__ dst,
                                                 const float* __restrict__ src,
                                                 const int nelems) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) dst[i] = __nv_cvt_float_to_fp8(src[i], __NV_SATFINITE, __NV_E4M3);
}

extern "C" __global__ void fp32_cvt_e5m2_kernel(uint8_t* __restrict__ dst,
                                                 const float* __restrict__ src,
                                                 const int nelems) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) dst[i] = __nv_cvt_float_to_fp8(src[i], __NV_SATFINITE, __NV_E5M2);
}
