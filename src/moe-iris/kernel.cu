#include <cfloat>
#include <cub/cub.cuh>
#include <cub/util_type.cuh>

struct Sum {
  template <typename T, typename U>
  __device__ __forceinline__ auto operator()(T&& t, U&& u) const
      -> decltype(std::forward<T>(t) + std::forward<U>(u))
  {
    return std::forward<T>(t) + std::forward<U>(u);
  }
};

struct Max {
  template <typename T, typename U>
  __device__ __forceinline__ typename std::common_type<T, U>::type
  operator()(T&& t, U&& u) const
  {
    return ((t) > (u)) ? (t) : (u);
  }
};

extern "C" __global__ void moeSoftmax_256(
    const float* __restrict__ input,
    float* output,
    const int num_cols)
{
  constexpr int TPB = 256;
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;
  __shared__ float normalizing_factor;
  __shared__ float float_max;
  const int thread_row_offset = blockIdx.x * num_cols;
  float threadData(-FLT_MAX);
  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData = fmaxf(input[idx], threadData);
  }
  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, Max());
  if (threadIdx.x == 0) float_max = maxElem;
  __syncthreads();
  threadData = 0;
  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData += expf(input[idx] - float_max);
  }
  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, Sum());
  if (threadIdx.x == 0) normalizing_factor = 1.f / Z;
  __syncthreads();
  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    output[idx] = expf(input[idx] - float_max) * normalizing_factor;
  }
}

extern "C" __global__ void moeTopK_256(
    const float* __restrict__ inputs_after_softmax,
    float* __restrict__ output,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert)
{
  constexpr int TPB = 256;
  using cub_kvp = cub::KeyValuePair<int, float>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;
  cub_kvp thread_kvp;
  cub::ArgMax arg_max;
  const int num_tokens = gridDim.x;
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int thread_read_offset = token * num_experts;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = -1.f;
    cub_kvp inp_kvp;
    for (int expert = tid; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];
      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * token + prior_k];
        if (prior_winning_expert == expert) inp_kvp = thread_kvp;
      }
      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }
    const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (tid == 0) {
      const int expert = result_kvp.key;
      const bool node_uses_expert = expert >= start_expert && expert < end_expert;
      const int idx = k * token + k_idx;
      output[idx] = result_kvp.value;
      indices[idx] = node_uses_expert ? (expert - start_expert) : num_experts;
      source_rows[idx] = k_idx * num_tokens + token;
    }
    __syncthreads();
  }
}
