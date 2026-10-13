#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "cpu_reference.h"
#include "utils.h"

using namespace flashinfer;

template <typename T>
const char* kernel_name();

template <>
const char* kernel_name<half>() {
  return "attention_multi_head_kv_cache_fp16";
}

template <>
const char* kernel_name<__hip_bfloat16>() {
  return "attention_multi_head_kv_cache_bf16";
}

template <typename DTypeQO, typename DTypeKV>
void test_decoding_kernel(iris::Platform& platform, int repeat, size_t num_qo_heads,
                          size_t num_kv_heads, size_t seq_len, size_t head_dim) {
  const size_t qo_elements = num_qo_heads * head_dim;
  const size_t kv_elements = seq_len * num_kv_heads * head_dim;
  std::vector<DTypeQO> q_host(qo_elements), o_host(qo_elements);
  std::vector<DTypeKV> k_host(kv_elements), v_host(kv_elements);
  utils::vec_normal_(q_host);
  utils::vec_normal_(k_host);
  utils::vec_normal_(v_host);

  const auto reference = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
      q_host, k_host, v_host, 1, seq_len, num_qo_heads, num_kv_heads, head_dim, false,
      QKVLayout::kNHD, PosEncodingMode::kNone);

  iris::DMem q_mem(q_host.data(), q_host.size() * sizeof(DTypeQO));
  iris::DMem k_mem(k_host.data(), k_host.size() * sizeof(DTypeKV));
  iris::DMem v_mem(v_host.data(), v_host.size() * sizeof(DTypeKV));
  iris::DMem o_mem(o_host.data(), o_host.size() * sizeof(DTypeQO));
  q_mem.update(q_host.data());
  k_mem.update(k_host.data());
  v_mem.update(v_host.data());

  unsigned int heads_q = static_cast<unsigned int>(num_qo_heads);
  unsigned int heads_kv = static_cast<unsigned int>(num_kv_heads);
  unsigned int length = static_cast<unsigned int>(seq_len);
  unsigned int dimension = static_cast<unsigned int>(head_dim);
  size_t gws[] = {qo_elements};
  size_t lws[] = {256};
  void* params[] = {&o_mem, &q_mem, &k_mem, &v_mem, &heads_q, &heads_kv, &length, &dimension};
  int pinfo[] = {iris_w, iris_r, iris_r, iris_r, sizeof(heads_q), sizeof(heads_kv),
                 sizeof(length), sizeof(dimension)};

  iris::Task task;
  task.kernel(kernel_name<DTypeQO>(), 1, nullptr, gws, lws, 8, params, pinfo);
  task.flush_out(o_mem);
  const int submit_status = task.submit(iris_gpu, nullptr, true);
  const int fetch_status = o_mem.fetch(o_host.data());
  if (submit_status != 0 || fetch_status != 0) {
    std::cerr << "IRIS submit_status=" << submit_status
              << " fetch_status=" << fetch_status << std::endl;
  }

  size_t errors = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < qo_elements; ++i) {
    const float actual = static_cast<float>(o_host[i]);
    const float expected = static_cast<float>(reference[i]);
    nan_detected |= std::isnan(actual);
    errors += !utils::isclose(actual, expected, 1e-2f, 1e-2f);
  }
  const float accuracy = 1.f - static_cast<float>(errors) / static_cast<float>(qo_elements);
  std::cout << "num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
            << ", seq_len=" << seq_len << ", head_dim=" << head_dim
            << ", kv_layout=NHD, pos_encoding_mode=None"
            << ", result accuracy (atol=1e-2, rtol=1e-2)=" << accuracy
            << (accuracy < 0.90f ? " FAIL " : " PASS ") << std::endl;
  if (nan_detected) std::cerr << "NaN detected." << std::endl;

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task repeat_task;
    repeat_task.kernel(kernel_name<DTypeQO>(), 1, nullptr, gws, lws, 8, params, pinfo);
    repeat_task.submit(iris_gpu, nullptr, true);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of SingleDecodeWithKVCache: %f (us)\n",
              (elapsed * 1e-3) / repeat);
}

template <typename DTypeQO, typename DTypeKV>
void single_decode_kernel(iris::Platform& platform, int repeat) {
  for (size_t seq_len : {size_t(1), size_t(3)}) {
    test_decoding_kernel<DTypeQO, DTypeKV>(platform, repeat, 32, 4, seq_len, 64);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  std::cout << "FlashInfer: SingleDecodeKernel (FP16)" << std::endl;
  single_decode_kernel<half, half>(platform, repeat);
  std::cout << "\nFlashInfer: SingleDecodeKernel (BF16)" << std::endl;
  single_decode_kernel<__hip_bfloat16, __hip_bfloat16>(platform, repeat);
  platform.finalize();
  return 0;
}
