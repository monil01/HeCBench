#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>

#include "reference.h"

static void SubmitGelu(const char* kernel_name, iris::DMem& src_mem, iris::DMem& bias_mem,
                       int width, int height, int batch_size, int block_size, bool flush)
{
  size_t gws[2] = {static_cast<size_t>(height) * static_cast<size_t>(block_size),
                   static_cast<size_t>(batch_size)};
  size_t lws[2] = {static_cast<size_t>(block_size), 1};
  iris::Task task;
  void* params[4] = {&src_mem, &bias_mem, &width, &height};
  int pinfo[4] = {iris_rw, iris_r, sizeof(width), sizeof(height)};
  task.kernel(kernel_name, 2, nullptr, gws, lws, 4, params, pinfo);
  if (flush) task.flush_out(src_mem);
  task.submit(iris_gpu, nullptr, 1);
}

static bool CompareHalf(const __half* ref, const __half* got, size_t count)
{
  for (size_t i = 0; i < count; i++) {
    if (std::fabs(__half2float(ref[i]) - __half2float(got[i])) > 1e-3f) {
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <batch> <sequence length> <hidden dimension> <repeat>\n", argv[0]);
    std::printf("The hidden dimension is a multiple of two\n");
    return 1;
  }

  const int batch_size = std::atoi(argv[1]);
  const int seq_len = std::atoi(argv[2]);
  const int hidden_dim = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);
  if (batch_size <= 0 || seq_len <= 0 || hidden_dim <= 0 || repeat <= 0 ||
      (hidden_dim % 2) != 0) {
    std::fprintf(stderr, "invalid arguments\n");
    return 1;
  }

  const size_t src_size = static_cast<size_t>(batch_size) * seq_len * hidden_dim;
  const size_t src_size_bytes = src_size * sizeof(__half);
  const size_t bias_size_bytes = static_cast<size_t>(hidden_dim) * sizeof(__half);

  std::srand(123);
  __half* input = static_cast<__half*>(std::malloc(src_size_bytes));
  __half* output = static_cast<__half*>(std::malloc(src_size_bytes));
  __half* output_ref = static_cast<__half*>(std::malloc(src_size_bytes));
  __half* bias = static_cast<__half*>(std::malloc(bias_size_bytes));
  if (input == nullptr || output == nullptr || output_ref == nullptr || bias == nullptr) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(input);
    std::free(output);
    std::free(output_ref);
    std::free(bias);
    return 1;
  }

  for (size_t i = 0; i < src_size; i++) {
    output_ref[i] = input[i] = __float2half(std::rand() / static_cast<float>(RAND_MAX));
  }
  for (int i = 0; i < hidden_dim; i++) {
    bias[i] = __float2half(-6 + (std::rand() % 12));
  }

  int block_size;
  if (hidden_dim >= 4096) {
    block_size = 512;
  } else if (hidden_dim >= 2048) {
    block_size = 256;
  } else {
    block_size = 128;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem output_mem(output, src_size_bytes);
  iris::DMem bias_mem(bias, bias_size_bytes);
  gelu_bias_loop_cpu(output_ref, bias, batch_size, hidden_dim, seq_len);

  std::memcpy(output, input, src_size_bytes);
  output_mem.update(output);
  bias_mem.update(bias);
  SubmitGelu("gelu_bias_loop_base", output_mem, bias_mem, hidden_dim, seq_len,
             batch_size, block_size, true);
  output_mem.fetch(output);
  bool ok = CompareHalf(output_ref, output, src_size);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::memcpy(output, input, src_size_bytes);
  output_mem.update(output);
  bias_mem.update(bias);
  SubmitGelu("gelu_bias_loop_base", output_mem, bias_mem, hidden_dim, seq_len,
             batch_size, block_size, true);
  output_mem.fetch(output);
  ok = CompareHalf(output_ref, output, src_size);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    SubmitGelu("gelu_bias_loop_base", output_mem, bias_mem, hidden_dim, seq_len,
               batch_size, block_size, false);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of vectorized kernel %f (ms)\n",
              (time * 1e-6f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    SubmitGelu("gelu_bias_loop_base", output_mem, bias_mem, hidden_dim, seq_len,
               batch_size, block_size, false);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of baseline kernel %f (ms)\n",
              (time * 1e-6f) / repeat);

  platform.finalize();
  std::free(input);
  std::free(output);
  std::free(output_ref);
  std::free(bias);
  return 0;
}
