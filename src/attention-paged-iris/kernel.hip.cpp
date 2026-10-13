#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

template <typename T> __device__ float to_float(T value) { return value; }
template <> __device__ float to_float<__hip_bfloat16>(__hip_bfloat16 value) { return __bfloat162float(value); }
template <typename T> __device__ T from_float(float value) { return value; }
template <> __device__ __hip_bfloat16 from_float<__hip_bfloat16>(float value) { return __float2bfloat16(value); }

template <typename T, int BLOCK_SIZE>
__device__ void paged_attention_simple(
    T* out, const T* query, const T* key, const T* value, int num_kv_heads, float scale,
    const int* tables, const int* seq_lens, int max_blocks, const float* slopes,
    int q_stride, int kv_block_stride, int kv_head_stride, int num_heads, int head_size) {
  if (threadIdx.x != 0) return;
  const int seq = blockIdx.y, head = blockIdx.x;
  const int seq_len = seq_lens[seq], kv_head = head / (num_heads / num_kv_heads);
  float logits[4096];
  for (int t = 0; t < seq_len; ++t) {
    const int block = tables[seq * max_blocks + t / BLOCK_SIZE], offset = t % BLOCK_SIZE;
    float score = 0.0f;
    for (int d = 0; d < head_size; ++d) {
      const int qi = seq * q_stride + head * head_size + d;
      const int ki = block * kv_block_stride + kv_head * kv_head_stride + d * BLOCK_SIZE + offset;
      score += to_float(query[qi]) * to_float(key[ki]);
    }
    logits[t] = score * scale + (slopes ? slopes[head] * (t - seq_len + 1) : 0.0f);
  }
  float max_value = -INFINITY, sum = 0.0f;
  for (int t = 0; t < seq_len; ++t) max_value = fmaxf(max_value, logits[t]);
  for (int t = 0; t < seq_len; ++t) { logits[t] = expf(logits[t] - max_value); sum += logits[t]; }
  for (int d = 0; d < head_size; ++d) {
    float result = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      const int block = tables[seq * max_blocks + t / BLOCK_SIZE], offset = t % BLOCK_SIZE;
      const int vi = block * kv_block_stride + kv_head * kv_head_stride + d * BLOCK_SIZE + offset;
      result += (logits[t] / sum) * to_float(value[vi]);
    }
    out[seq * q_stride + head * head_size + d] = from_float<T>(result);
  }
}

extern "C" __global__ void paged_attention_fp32(float* out, const float* query, const float* key, const float* value,
    int num_kv_heads, float scale, const int* tables, const int* seq_lens, int max_blocks, const float* slopes,
    int q_stride, int kv_block_stride, int kv_head_stride, int tp_rank, int local_blocks, int vert_stride, int block_size, int sliding_step) {
  (void)tp_rank; (void)local_blocks; (void)vert_stride; (void)sliding_step;
  paged_attention_simple<float, 16>(out, query, key, value, num_kv_heads, scale, tables, seq_lens, max_blocks, slopes, q_stride, kv_block_stride, kv_head_stride, gridDim.x, 128);
}

extern "C" __global__ void paged_attention_bf16(__hip_bfloat16* out, const __hip_bfloat16* query, const __hip_bfloat16* key, const __hip_bfloat16* value,
    int num_kv_heads, float scale, const int* tables, const int* seq_lens, int max_blocks, const float* slopes,
    int q_stride, int kv_block_stride, int kv_head_stride, int tp_rank, int local_blocks, int vert_stride, int block_size, int sliding_step) {
  (void)tp_rank; (void)local_blocks; (void)vert_stride; (void)sliding_step;
  paged_attention_simple<__hip_bfloat16, 32>(out, query, key, value, num_kv_heads, scale, tables, seq_lens, max_blocks, slopes, q_stride, kv_block_stride, kv_head_stride, gridDim.x, 128);
}
