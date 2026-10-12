#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <limits>

inline __device__ float to_float(float u) { return u; }
inline __device__ float to_float(half u) { return __half2float(u); }
inline __device__ float to_float(__nv_bfloat16 u) { return __bfloat162float(u); }
inline __device__ void from_float(float& d, float s) { d = s; }
inline __device__ void from_float(half& d, float s) { d = __float2half(s); }
inline __device__ void from_float(__nv_bfloat16& d, float s) { d = __float2bfloat16(s); }

template <typename scalar_t, const uint32_t NUM_THREADS>
__device__ void merge_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ prefix_output,
    const scalar_t* __restrict__ suffix_output,
    float* __restrict__ lse,
    const float* __restrict__ prefix_lse,
    const float* __restrict__ suffix_lse,
    const uint64_t num_tokens,
    const uint32_t num_heads,
    const uint32_t head_size)
{
  using pack_128b_t = uint4;
  const uint32_t pack_size = 16 / sizeof(scalar_t);
  const uint32_t threads_per_head = head_size / pack_size;
  const uint64_t global_idx = static_cast<uint64_t>(blockIdx.x) * NUM_THREADS + threadIdx.x;
  const uint64_t token_head_threads = num_tokens * num_heads * threads_per_head;
  if (global_idx >= token_head_threads) return;
  const uint64_t token_head_idx = global_idx / threads_per_head;
  const uint32_t pack_idx = global_idx % threads_per_head;
  const uint64_t token_idx = token_head_idx / num_heads;
  const uint32_t head_idx = token_head_idx % num_heads;
  const uint64_t lse_offset = token_idx * num_heads + head_idx;
  const uint64_t head_offset = lse_offset * head_size;
  const scalar_t* prefix_head_ptr = prefix_output + head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + head_offset;
  scalar_t* output_head_ptr = output + head_offset;
  float p_lse = prefix_lse[lse_offset];
  float s_lse = suffix_lse[lse_offset];
  p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;
  const float max_lse = fmaxf(p_lse, s_lse);
  const float p_se = expf(p_lse - max_lse);
  const float s_se = expf(s_lse - max_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;
  pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[pack_idx];
  pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(suffix_head_ptr)[pack_idx];
  pack_128b_t o_out_pack;
#pragma unroll
  for (uint32_t i = 0; i < pack_size; ++i) {
    const float p_out_f = to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
    const float s_out_f = to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
    from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i],
               p_out_f * p_scale + (s_out_f * s_scale));
  }
  reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_idx] = o_out_pack;
  if (pack_idx == 0) lse[lse_offset] = logf(out_se) + max_lse;
}

extern "C" __global__ void merge_attn_states_float(
    float* output, const float* prefix_output, const float* suffix_output,
    float* lse, const float* prefix_lse, const float* suffix_lse,
    const uint64_t num_tokens, const uint32_t num_heads, const uint32_t head_size)
{
  merge_impl<float, 128>(output, prefix_output, suffix_output, lse, prefix_lse,
                         suffix_lse, num_tokens, num_heads, head_size);
}

extern "C" __global__ void merge_attn_states_half(
    half* output, const half* prefix_output, const half* suffix_output,
    float* lse, const float* prefix_lse, const float* suffix_lse,
    const uint64_t num_tokens, const uint32_t num_heads, const uint32_t head_size)
{
  merge_impl<half, 128>(output, prefix_output, suffix_output, lse, prefix_lse,
                        suffix_lse, num_tokens, num_heads, head_size);
}

extern "C" __global__ void merge_attn_states_bfloat16(
    __nv_bfloat16* output, const __nv_bfloat16* prefix_output,
    const __nv_bfloat16* suffix_output, float* lse, const float* prefix_lse,
    const float* suffix_lse, const uint64_t num_tokens, const uint32_t num_heads,
    const uint32_t head_size)
{
  merge_impl<__nv_bfloat16, 128>(output, prefix_output, suffix_output, lse,
                                 prefix_lse, suffix_lse, num_tokens,
                                 num_heads, head_size);
}
