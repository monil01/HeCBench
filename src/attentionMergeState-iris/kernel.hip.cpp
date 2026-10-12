#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float merge_lse(const float p, const float s,
                                           float* p_scale, float* s_scale) {
  const float pp = isinf(p) ? -INFINITY : p;
  const float ss = isinf(s) ? -INFINITY : s;
  const float m = fmaxf(pp, ss);
  const float pe = expf(pp - m);
  const float se = expf(ss - m);
  const float sum = pe + se;
  *p_scale = pe / sum;
  *s_scale = se / sum;
  return logf(sum) + m;
}

template <typename T>
__device__ __forceinline__ float as_float(T value) { return value; }
template <>
__device__ __forceinline__ float as_float<half>(half value) {
  return __half2float(value);
}
template <>
__device__ __forceinline__ float as_float<__hip_bfloat16>(__hip_bfloat16 value) {
  return __bfloat162float(value);
}
template <typename T>
__device__ __forceinline__ T from_float(float value) { return value; }
template <>
__device__ __forceinline__ half from_float<half>(float value) {
  return __float2half(value);
}
template <>
__device__ __forceinline__ __hip_bfloat16 from_float<__hip_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T>
__device__ void merge_attn_states_impl(
    T* output, const T* prefix_output, const T* suffix_output,
    float* lse, const float* prefix_lse, const float* suffix_lse,
    uint64_t num_tokens, uint32_t num_heads, uint32_t head_size) {
  constexpr uint32_t kThreads = 128;
  const uint32_t pack_size = 16 / sizeof(T);
  const uint32_t threads_per_head = head_size / pack_size;
  const uint64_t global_idx = static_cast<uint64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const uint64_t total = num_tokens * num_heads * threads_per_head;
  if (global_idx >= total) return;

  const uint64_t token_head = global_idx / threads_per_head;
  const uint32_t pack_idx = global_idx % threads_per_head;
  const uint64_t token = token_head / num_heads;
  const uint32_t head = token_head % num_heads;
  const uint64_t lse_idx = token * num_heads + head;
  const uint64_t head_offset = lse_idx * head_size;
  const uint32_t pack_offset = pack_idx * pack_size;

  float p_scale, s_scale;
  const float out_lse = merge_lse(prefix_lse[lse_idx], suffix_lse[lse_idx],
                                  &p_scale, &s_scale);
  for (uint32_t i = 0; i < pack_size && pack_offset + i < head_size; ++i) {
    const uint64_t index = head_offset + pack_offset + i;
    output[index] = from_float<T>(as_float(prefix_output[index]) * p_scale +
                                  as_float(suffix_output[index]) * s_scale);
  }
  if (pack_idx == 0) lse[lse_idx] = out_lse;
}

extern "C" __global__ void merge_attn_states_fp32(
    float* o, const float* p, const float* s, float* l,
    const float* pl, const float* sl, uint64_t n, uint32_t h, uint32_t d) {
  merge_attn_states_impl<float>(o, p, s, l, pl, sl, n, h, d);
}

extern "C" __global__ void merge_attn_states_fp16(
    half* o, const half* p, const half* s, float* l,
    const float* pl, const float* sl, uint64_t n, uint32_t h, uint32_t d) {
  merge_attn_states_impl<half>(o, p, s, l, pl, sl, n, h, d);
}

extern "C" __global__ void merge_attn_states_bf16(
    __hip_bfloat16* o, const __hip_bfloat16* p, const __hip_bfloat16* s, float* l,
    const float* pl, const float* sl, uint64_t n, uint32_t h, uint32_t d) {
  merge_attn_states_impl<__hip_bfloat16>(o, p, s, l, pl, sl, n, h, d);
}
