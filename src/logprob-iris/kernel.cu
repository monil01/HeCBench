#include <cfloat>

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  val = warpReduceSum<T>(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : static_cast<T>(0.0f);
  return warpReduceSum<T>(val);
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  val = warpReduceMax(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  return warpReduceMax(val);
}

extern "C" __global__ void log_probs_kernel_float(
    float* log_probs,
    const float* logits,
    const int* ids,
    const int* lengths,
    const int max_input_length,
    const int batch_size,
    const int vocab_size,
    const int vocab_size_padded)
{
  int tidx = threadIdx.x;
  int step = blockIdx.x;
  int bidx = blockIdx.y;
  __shared__ float s_max_logit;

  if (bidx < batch_size && step < lengths[bidx] - 1) {
    int step_offset = step * vocab_size_padded;
    int batch_offset = bidx * max_input_length * vocab_size_padded;
    logits += step_offset + batch_offset;
    float local_max = -FLT_MAX;
    for (int i = tidx; i < vocab_size; i += blockDim.x) {
      local_max = fmaxf(local_max, logits[i]);
    }
    float max_val = blockReduceMax<float>(local_max);
    if (tidx == 0) s_max_logit = max_val;
    __syncthreads();
    float local_sum_exp = 0.0f;
    for (int i = tidx; i < vocab_size; i += blockDim.x) {
      local_sum_exp += expf(logits[i] - s_max_logit);
    }
    float sum_exp = blockReduceSum<float>(local_sum_exp);
    if (tidx == 0) {
      int idx = step + bidx * (max_input_length - 1);
      int token_idx = step + 1 + bidx * max_input_length;
      log_probs[idx] = logits[ids[token_idx]] - s_max_logit -
                       logf(sum_exp + 1e-9f);
    }
  }
}

extern "C" __global__ void accumulate_log_probs(
    float* cum_log_probs,
    const float* log_probs,
    const int* lengths,
    const int max_input_length,
    const int batch_size)
{
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  if (bidx >= batch_size) return;
  int length = lengths[bidx];
  log_probs += bidx * (max_input_length - 1);
  float local_accum = 0.0f;
  for (int step = tidx; step < length - 1; step += blockDim.x) {
    local_accum += log_probs[step];
  }
  float accum = blockReduceSum<float>(local_accum);
  if (tidx == 0) cum_log_probs[bidx] = accum;
}
