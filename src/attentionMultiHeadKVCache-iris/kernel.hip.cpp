#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

template <typename T>
__device__ __forceinline__ float as_float(T value);

template <>
__device__ __forceinline__ float as_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float as_float<__hip_bfloat16>(__hip_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __hip_bfloat16 from_float<__hip_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T>
__device__ void attention_kernel(const T* q, const T* k, const T* v, T* o,
                                 unsigned int num_qo_heads, unsigned int num_kv_heads,
                                 unsigned int seq_len, unsigned int head_dim) {
  const size_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t output_size = static_cast<size_t>(num_qo_heads) * head_dim;
  if (output_index >= output_size) return;

  const unsigned int qo_head = output_index / head_dim;
  const unsigned int feature = output_index % head_dim;
  const unsigned int kv_head = qo_head / (num_qo_heads / num_kv_heads);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float logits[64];
  float max_logit = -3.402823466e+38F;
  for (unsigned int kv_index = 0; kv_index < seq_len; ++kv_index) {
    float dot = 0.f;
    for (unsigned int d = 0; d < head_dim; ++d) {
      const size_t q_index = static_cast<size_t>(qo_head) * head_dim + d;
      const size_t kv_offset = (static_cast<size_t>(kv_index) * num_kv_heads + kv_head) * head_dim;
      dot += as_float(q[q_index]) * as_float(k[kv_offset + d]);
    }
    logits[kv_index] = dot * scale;
    max_logit = fmaxf(max_logit, logits[kv_index]);
  }

  float denominator = 0.f;
  for (unsigned int kv_index = 0; kv_index < seq_len; ++kv_index) {
    logits[kv_index] = expf(logits[kv_index] - max_logit);
    denominator += logits[kv_index];
  }

  float result = 0.f;
  for (unsigned int kv_index = 0; kv_index < seq_len; ++kv_index) {
    const size_t kv_offset = (static_cast<size_t>(kv_index) * num_kv_heads + kv_head) * head_dim;
    result += (logits[kv_index] / denominator) * as_float(v[kv_offset + feature]);
  }
  o[output_index] = from_float<T>(result);
}

}  // namespace

extern "C" __global__ void attention_multi_head_kv_cache_fp16(
    const __half* q, const __half* k, const __half* v, __half* o,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int seq_len,
    unsigned int head_dim) {
  attention_kernel(q, k, v, o, num_qo_heads, num_kv_heads, seq_len, head_dim);
}

extern "C" __global__ void attention_multi_head_kv_cache_bf16(
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v,
    __hip_bfloat16* o, unsigned int num_qo_heads, unsigned int num_kv_heads,
    unsigned int seq_len, unsigned int head_dim) {
  attention_kernel(q, k, v, o, num_qo_heads, num_kv_heads, seq_len, head_dim);
}
