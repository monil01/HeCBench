#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void moeSoftmax_reference(const float* input, float* output,
                                 int num_tokens, int num_cols)
{
  for (int token = 0; token < num_tokens; token++) {
    const int row_offset = token * num_cols;
    float max_val = -FLT_MAX;
    for (int ii = 0; ii < num_cols; ii++) {
      max_val = std::fmax(input[row_offset + ii], max_val);
    }
    float sum = 0.0f;
    for (int ii = 0; ii < num_cols; ii++) {
      sum += std::exp(input[row_offset + ii] - max_val);
    }
    const float normalizing_factor = 1.f / sum;
    for (int ii = 0; ii < num_cols; ii++) {
      output[row_offset + ii] =
          std::exp(input[row_offset + ii] - max_val) * normalizing_factor;
    }
  }
}

struct Kvp {
  int key;
  float value;
};

static Kvp arg_max(const Kvp& a, const Kvp& b)
{
  if (a.value > b.value) return {a.key, a.value};
  if (a.value == b.value) return {std::min(a.key, b.key), a.value};
  return {b.key, b.value};
}

static void moeTopK_reference(const float* inputs_after_softmax,
                              float* weights,
                              int* indices,
                              int* source_rows,
                              int num_tokens,
                              int num_experts,
                              int k,
                              int start_expert,
                              int end_expert)
{
  for (int token = 0; token < num_tokens; token++) {
    const int read_offset = token * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      Kvp thread_kvp{0, -1.f};
      for (int expert = 0; expert < num_experts; expert++) {
        Kvp inp_kvp{expert, inputs_after_softmax[read_offset + expert]};
        for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
          const int prior_winning_expert = indices[k * token + prior_k];
          if (prior_winning_expert == expert) inp_kvp = thread_kvp;
        }
        thread_kvp = arg_max(inp_kvp, thread_kvp);
      }
      const int expert = thread_kvp.key;
      const bool node_uses_expert =
          expert >= start_expert && expert < end_expert;
      const int idx = k * token + k_idx;
      weights[idx] = thread_kvp.value;
      indices[idx] = node_uses_expert ? (expert - start_expert) : num_experts;
      source_rows[idx] = k_idx * num_tokens + token;
    }
  }
}

static void run_kernel(const char* name, size_t gws, size_t lws, int nparams,
                       void** params, int* pinfo, iris::DMem* flush0,
                       iris::DMem* flush1, iris::DMem* flush2)
{
  iris::Task task;
  size_t global[1] = {gws};
  size_t local[1] = {lws};
  task.kernel(name, 1, nullptr, global, local, nparams, params, pinfo);
  if (flush0) task.flush_out(*flush0);
  if (flush1) task.flush_out(*flush1);
  if (flush2) task.flush_out(*flush2);
  task.submit(iris_gpu, nullptr, 1);
}

static int topk_softmax(int num_tokens, int num_experts, int topk, int repeat)
{
  const int index_size = num_tokens * topk;
  const int weight_size = num_tokens * topk;
  const int output_size = num_tokens * num_experts;
  const size_t index_size_bytes = index_size * sizeof(int);
  const size_t weight_size_bytes = weight_size * sizeof(float);
  const size_t output_size_bytes = output_size * sizeof(float);

  std::vector<float> topk_weights(weight_size), topk_weights_ref(weight_size);
  std::vector<int> topk_indices(index_size), topk_indices_ref(index_size);
  std::vector<int> token_expert_indices(index_size);
  std::vector<int> token_expert_indices_ref(index_size);
  std::vector<float> gating_output(output_size);
  std::vector<float> softmax_workspace(output_size);

  std::mt19937 gen(19937);
  std::uniform_int_distribution<> distrib(-100000, 100000);
  for (int i = 0; i < output_size; i++) gating_output[i] = distrib(gen);
  for (int i = 0; i < topk; i++) {
    for (int j = 0; j < num_tokens; j++) {
      topk_indices_ref[i * num_tokens + j] =
          topk_indices[i * num_tokens + j] =
              std::abs(distrib(gen)) % num_experts;
    }
  }

  moeSoftmax_reference(gating_output.data(), softmax_workspace.data(),
                       num_tokens, num_experts);
  moeTopK_reference(softmax_workspace.data(), topk_weights_ref.data(),
                    topk_indices_ref.data(), token_expert_indices_ref.data(),
                    num_tokens, num_experts, topk, 0, num_experts);

  std::vector<float> d_softmax_workspace(output_size);
  iris::DMem mem_topk_weights(topk_weights.data(), weight_size_bytes);
  iris::DMem mem_topk_indices(topk_indices.data(), index_size_bytes);
  iris::DMem mem_token_expert_indices(token_expert_indices.data(),
                                      index_size_bytes);
  iris::DMem mem_gating_output(gating_output.data(), output_size_bytes);
  iris::DMem mem_softmax_workspace(d_softmax_workspace.data(),
                                   output_size_bytes);

  constexpr int TPB = 256;
  void* softmax_params[3] = {&mem_gating_output, &mem_softmax_workspace,
                             &num_experts};
  int softmax_pinfo[3] = {iris_r, iris_w, sizeof(num_experts)};
  void* topk_params[8] = {&mem_softmax_workspace, &mem_topk_weights,
                          &mem_topk_indices, &mem_token_expert_indices,
                          &num_experts, &topk, nullptr, &num_experts};
  int start_expert = 0;
  topk_params[6] = &start_expert;
  int topk_pinfo[8] = {iris_r, iris_w, iris_rw, iris_w, sizeof(num_experts),
                       sizeof(topk), sizeof(start_expert),
                       sizeof(num_experts)};

  run_kernel("moeSoftmax_256", static_cast<size_t>(num_tokens * TPB), TPB,
             3, softmax_params, softmax_pinfo, nullptr, nullptr, nullptr);
  run_kernel("moeTopK_256", static_cast<size_t>(num_tokens * TPB), TPB,
             8, topk_params, topk_pinfo, &mem_topk_weights, &mem_topk_indices,
             &mem_token_expert_indices);

  mem_topk_weights.fetch(topk_weights.data());
  mem_topk_indices.fetch(topk_indices.data());
  mem_token_expert_indices.fetch(token_expert_indices.data());

  int error = std::memcmp(topk_indices.data(), topk_indices_ref.data(),
                          index_size_bytes);
  error += std::memcmp(token_expert_indices.data(),
                       token_expert_indices_ref.data(), index_size_bytes);
  for (int i = 0; i < weight_size; i++) {
    if (std::fabs(topk_weights[i] - topk_weights_ref[i]) > 1e-3f) {
      error = 1;
      break;
    }
  }
  std::printf("%s\n", error ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_kernel("moeSoftmax_256", static_cast<size_t>(num_tokens * TPB), TPB,
               3, softmax_params, softmax_pinfo, nullptr, nullptr, nullptr);
    run_kernel("moeTopK_256", static_cast<size_t>(num_tokens * TPB), TPB,
               8, topk_params, topk_pinfo, nullptr, nullptr, nullptr);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of kernels: %f (us)\n",
              (time * 1e-3f) / repeat);
  return error ? 1 : 0;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <number of tokens> <number of experts> <top K> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int num_tokens = std::atoi(argv[1]);
  const int num_experts = std::atoi(argv[2]);
  const int topk = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);
  int rc = topk_softmax(num_tokens, num_experts, topk, repeat);
  platform.finalize();
  return rc;
}
