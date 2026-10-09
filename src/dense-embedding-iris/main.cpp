#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

template <typename T>
static void reference(const T* input,
                      const T* dense,
                      T* output,
                      int embedding_dim,
                      int batch_size,
                      const int* offset)
{
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    const int range = offset[batch_idx + 1] - offset[batch_idx];
    for (int idx = 0; idx < embedding_dim; idx++) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        output[offset[batch_idx] + nested_idx] =
          input[offset[batch_idx] + nested_idx] + dense_elem;
      }
    }
  }
}

static double run_kernel(const char* kernel_name,
                         int batch_size,
                         int block_size,
                         int repeat,
                         int ncols,
                         iris::DMem& mem_input,
                         iris::DMem& mem_dense,
                         iris::DMem& mem_output,
                         iris::DMem& mem_offset)
{
  const size_t gws[1] = { static_cast<size_t>(batch_size) * static_cast<size_t>(block_size) };
  const size_t lws[1] = { static_cast<size_t>(block_size) };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[5] = { &mem_input, &mem_dense, &mem_output, &ncols, &mem_offset };
    int pinfo[5] = { iris_r, iris_r, iris_w, sizeof(ncols), iris_r };
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_output);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-3) / repeat;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of rows> <batch size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nrows = std::atoi(argv[1]);
  const int batch_size = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  assert(nrows > batch_size * batch_size);

  std::printf("Number of rows in the embedding table: %d\n", nrows);
  std::printf("Batch size: %d\n", batch_size);

  const int embed_dims[] = { 768, 2048, 12288 };
  bool all_ok = true;

  for (size_t n = 0; n < sizeof(embed_dims) / sizeof(int); n++) {
    int ncols = embed_dims[n];
    std::printf("\nEmbedding dimension: %d\n", ncols);

    int input_size = nrows * ncols;
    size_t input_size_bytes = static_cast<size_t>(input_size) * sizeof(float);
    int dense_size = batch_size * ncols;
    size_t dense_size_bytes = static_cast<size_t>(dense_size) * sizeof(float);
    size_t batch_size_bytes = static_cast<size_t>(batch_size + 1) * sizeof(int);

    std::vector<float> input(input_size);
    std::vector<float> dense(dense_size);
    std::vector<float> output_k1(input_size);
    std::vector<float> output_k2(input_size);
    std::vector<float> output_k3(input_size);
    std::vector<float> output_ref(input_size);
    std::vector<int> input_offset(batch_size + 1);

    std::srand(123);
    input_offset[0] = 0;
    for (int i = 1; i <= batch_size; i++) {
      input_offset[i] = input_offset[i - 1] + (std::rand() % batch_size + 1) * ncols;
    }

    std::default_random_engine g(123);
    std::uniform_real_distribution<float> distr(-1.f, 1.f);
    for (int i = 0; i < dense_size; i++) {
      dense[i] = distr(g);
    }

    for (int i = 0; i < input_size; i++) {
      input[i] = distr(g);
      output_ref[i] = 0.0f;
    }

    reference(input.data(), dense.data(), output_ref.data(), ncols, batch_size, input_offset.data());

    iris::DMem mem_input(input.data(), input_size_bytes);
    iris::DMem mem_dense(dense.data(), dense_size_bytes);
    iris::DMem mem_output(output_k1.data(), input_size_bytes);
    iris::DMem mem_offset(input_offset.data(), batch_size_bytes);

    for (int block_size = 128; block_size <= 1024; block_size *= 2) {
      std::printf("block size: %d\n", block_size);

      std::fill(output_k1.begin(), output_k1.end(), 0.0f);
      mem_output.update(output_k1.data());
      double us = run_kernel("dense_esuhm", batch_size, block_size, repeat, ncols,
                             mem_input, mem_dense, mem_output, mem_offset);
      std::printf("Average execution time of dense embedding kernel (k1): %f (us)\n", us);
      mem_output.fetch(output_k1.data());

      std::fill(output_k2.begin(), output_k2.end(), 0.0f);
      mem_output.update(output_k2.data());
      us = run_kernel("dense_esuhm2", batch_size, block_size, repeat, ncols,
                      mem_input, mem_dense, mem_output, mem_offset);
      std::printf("Average execution time of dense embedding kernel (k2): %f (us)\n", us);
      mem_output.fetch(output_k2.data());

      std::fill(output_k3.begin(), output_k3.end(), 0.0f);
      mem_output.update(output_k3.data());
      us = run_kernel("dense_esuhm3", batch_size, block_size, repeat, ncols,
                      mem_input, mem_dense, mem_output, mem_offset);
      std::printf("Average execution time of dense embedding kernel (k3): %f (us)\n", us);
      mem_output.fetch(output_k3.data());

      bool ok = true;
      for (int i = 0; i < input_size; i++) {
        if (std::fabs(output_k1[i] - output_ref[i]) > 1e-3f ||
            std::fabs(output_k2[i] - output_ref[i]) > 1e-3f ||
            std::fabs(output_k3[i] - output_ref[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
      std::printf("%s\n", ok ? "PASS" : "FAIL");
      all_ok = all_ok && ok;
    }
  }

  platform.finalize();
  return all_ok ? 0 : 1;
}
