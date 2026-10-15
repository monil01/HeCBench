#include <iris/iris.hpp>

#include <hip/hip_fp16.h>
#include <hip/hip_vector_types.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static int ceildiv(int a, int b) { return (a + b - 1) / b; }

static void code1x16_matvec(
    iris::DMem& mem_codes,
    iris::DMem& mem_input,
    iris::DMem& mem_output,
    iris::DMem& mem_codebook,
    int prob_m,
    int prob_k,
    int4 codebook_a_sizes,
    int codebook_stride,
    int vector_index,
    bool flush_output)
{
  const int thread_m = 16;
  const int blocks = ceildiv(prob_m, thread_m);
  const int threads = 64 * thread_m;
  size_t gws[1] = { static_cast<size_t>(blocks * threads) };
  size_t lws[1] = { static_cast<size_t>(threads) };

  iris::Task task;
  void* params[9] = {
    &mem_codes,
    &mem_input,
    &mem_output,
    &mem_codebook,
    &prob_m,
    &prob_k,
    &codebook_a_sizes,
    &codebook_stride,
    &vector_index
  };
  int pinfo[9] = {
    iris_r,
    iris_r,
    iris_rw,
    iris_r,
    sizeof(prob_m),
    sizeof(prob_k),
    sizeof(codebook_a_sizes),
    sizeof(codebook_stride),
    sizeof(vector_index)
  };

  auto start = std::chrono::steady_clock::now();
  task.kernel("Code1x16MatVec64", 1, nullptr, gws, lws, 9, params, pinfo);
  if (flush_output) {
    task.flush_out(mem_output);
  }
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("kernel execution time: %f (us)\n", (time * 1e-3f) / 1);
}

#include "verify_output.h"

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int b = 4;
  const size_t input_size = b * 4096;
  const size_t output_size = b * 12288;
  const size_t rows = 512;
  const size_t cols = 12288;
  const size_t code_size = cols * rows;
  const size_t entries = 65536;
  const size_t codebook_size = 3 * entries * 8;
  const int4 codebook_a_sizes = make_int4(4096, 8192, 12288, 122880);
  const int codebook_stride = entries * 8 * sizeof(__half) / sizeof(int4);

  const size_t input_size_bytes = input_size * sizeof(__half);
  const size_t output_size_bytes = output_size * sizeof(__half);
  const size_t code_size_bytes = code_size * sizeof(short);
  const size_t codebook_size_bytes = codebook_size * sizeof(__half);

  __half* h_input = static_cast<__half*>(std::malloc(input_size_bytes));
  __half* h_output = static_cast<__half*>(std::malloc(output_size_bytes));
  short* h_codes = static_cast<short*>(std::malloc(code_size_bytes));
  __half* h_codebook = static_cast<__half*>(std::malloc(codebook_size_bytes));
  if (!h_input || !h_output || !h_codes || !h_codebook) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_input);
    std::free(h_output);
    std::free(h_codes);
    std::free(h_codebook);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (size_t i = 0; i < input_size; i++) {
    h_input[i] = static_cast<__half>(static_cast<float>(i) / input_size);
  }
  for (size_t i = 0; i < code_size; i++) {
    h_codes[i] = static_cast<short>(std::rand() % 65536 - 32768);
  }
  for (size_t i = 0; i < codebook_size; i++) {
    h_codebook[i] = static_cast<__half>(static_cast<float>(i) / codebook_size);
  }
  for (size_t i = 0; i < output_size; i++) {
    h_output[i] = static_cast<__half>(0.0f);
  }

  iris::DMem mem_input(h_input, input_size_bytes);
  iris::DMem mem_output(h_output, output_size_bytes);
  iris::DMem mem_codes(h_codes, code_size_bytes);
  iris::DMem mem_codebook(h_codebook, codebook_size_bytes);

  const int prob_m = 12288;
  const int prob_k = 4096;
  for (int i = 0; i < b; ++i) {
    code1x16_matvec(
        mem_codes,
        mem_input,
        mem_output,
        mem_codebook,
        prob_m,
        prob_k,
        codebook_a_sizes,
        codebook_stride,
        i,
        i == b - 1);
  }
  mem_output.fetch(h_output);
  verify_output(h_input, h_codes, h_codebook, h_output, b);

#ifdef DEBUG
  for (int i = 0; i < b; ++i) {
    __half* h_output_vec = h_output + 12288 * i;
    for (int j = 0; j < 12288; j++) {
      std::printf("row=%d col=%d val=%f\n", i, j, __half2float(h_output_vec[j]));
    }
  }
#endif

  std::free(h_input);
  std::free(h_output);
  std::free(h_codes);
  std::free(h_codebook);
  platform.finalize();
  return 0;
}
