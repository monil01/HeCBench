#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cstdint>

__device__ inline float fp32_from_bits(uint32_t w) {
  return __uint_as_float(static_cast<unsigned int>(w));
}

__device__ inline uint32_t fp32_to_bits(float f) {
  return static_cast<uint32_t>(__float_as_uint(f));
}

inline __device__ uint8_t fp8e5m2fnuz_from_fp32_value(float f) {
  constexpr uint32_t fnuz_max = UINT32_C(0x8F) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(0x85) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint32_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fnuz_max) {
    return 0x80;
  }

  if (f_bits < (UINT32_C(0x70) << 23)) {
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
    if (result == 0) {
      return 0;
    }
  } else {
    uint8_t mant_odd = (f_bits >> 21) & 1;
    f_bits += (static_cast<uint32_t>(16 - 127) << 23) + 0xFFFFF;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 21);
  }

  result |= sign >> 24;
  return static_cast<uint8_t>(result);
}

inline __device__ uint8_t fp8e4m3fnuz_from_fp32_value(float f) {
  constexpr uint32_t fnuz_max = UINT32_C(0x87) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(0x8C) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint32_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fnuz_max) {
    return 0x80;
  }

  if (f_bits < (UINT32_C(0x78) << 23)) {
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
    if (result == 0) {
      return 0;
    }
  } else {
    uint8_t mant_odd = (f_bits >> 20) & 1;
    f_bits += (static_cast<uint32_t>(8 - 127) << 23) + 0x7FFFF;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }

  result |= sign >> 24;
  return static_cast<uint8_t>(result);
}

extern "C" __global__ void ref_fp32_cvt_e5m2fnuz_kernel(
    uint8_t* __restrict__ dst, const float* __restrict__ src, const int nelems) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = fp8e5m2fnuz_from_fp32_value(src[i]);
  }
}

extern "C" __global__ void ref_fp32_cvt_e4m3fnuz_kernel(
    uint8_t* __restrict__ dst, const float* __restrict__ src, const int nelems) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = fp8e4m3fnuz_from_fp32_value(src[i]);
  }
}

extern "C" __global__ void fp32_cvt_e4m3_kernel(uint8_t* __restrict__ dst,
                                                 const float* __restrict__ src,
                                                 const int nelems) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = __hip_cvt_float_to_fp8(src[i], __HIP_SATFINITE, __HIP_E4M3_FNUZ);
  }
}

extern "C" __global__ void fp32_cvt_e5m2_kernel(uint8_t* __restrict__ dst,
                                                 const float* __restrict__ src,
                                                 const int nelems) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = __hip_cvt_float_to_fp8(src[i], __HIP_SATFINITE, __HIP_E5M2_FNUZ);
  }
}
