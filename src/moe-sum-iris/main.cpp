#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void moe_sum_ref(int topk, float* out, const float* input, int num_tokens, int hidden_size)
{
  for (int block = 0; block < num_tokens; ++block) {
    const int64_t output_base = static_cast<int64_t>(block) * hidden_size;
    const int64_t input_base = output_base * topk;
    for (int idx = 0; idx < hidden_size; ++idx) {
      float x = 0.0f;
      for (int k = 0; k < topk; ++k) {
        x += input[input_base + k * hidden_size + idx];
      }
      out[output_base + idx] = x;
    }
  }
}

static const char* scalar_kernel_name(int topk)
{
  switch (topk) {
    case 2: return "moe_sum_kernel_topk2";
    case 3: return "moe_sum_kernel_topk3";
    case 4: return "moe_sum_kernel_topk4";
    default: return nullptr;
  }
}

static const char* vec4_kernel_name(int topk)
{
  switch (topk) {
    case 2: return "moe_sum_kernel_vec4_topk2";
    case 3: return "moe_sum_kernel_vec4_topk3";
    case 4: return "moe_sum_kernel_vec4_topk4";
    default: return nullptr;
  }
}

static int64_t run_kernel(const char* kernel_name,
                          iris::DMem& mem_input,
                          iris::DMem& mem_output,
                          int hidden_size,
                          int num_tokens,
                          int repeat,
                          bool vec4)
{
  const int block = std::min(vec4 ? hidden_size / 4 : hidden_size, 1024);
  size_t gws[1] = { static_cast<size_t>(num_tokens) * block };
  size_t lws[1] = { static_cast<size_t>(block) };

  for (int i = 0; i < 100; i++) {
    iris::Task task;
    void* params[3] = { &mem_output, &mem_input, &hidden_size };
    int pinfo[3] = { iris_w, iris_r, sizeof(hidden_size) };
    task.kernel(kernel_name, 1, nullptr, gws, lws, 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = { &mem_output, &mem_input, &hidden_size };
    int pinfo[3] = { iris_w, iris_r, sizeof(hidden_size) };
    task.kernel(kernel_name, 1, nullptr, gws, lws, 3, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_output);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of tokens> <hidden size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int num_tokens = std::atoi(argv[1]);
  const int hidden_size = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  if (hidden_size % 4) {
    std::printf("Hidden size is a multiple of four\n");
    platform.finalize();
    return 1;
  }

  const int64_t output_elems = static_cast<int64_t>(num_tokens) * hidden_size;
  const int64_t output_size_bytes = output_elems * sizeof(float);
  std::vector<float> r_output(output_elems);
  std::vector<float> output(output_elems);
  std::vector<float> output_vec4(output_elems);

  int failures = 0;
  for (int topk = 2; topk <= 4; topk++) {
    const int64_t input_elems = output_elems * topk;
    const int64_t input_size_bytes = input_elems * sizeof(float);
    std::vector<float> input(input_elems);

    std::mt19937 gen(topk);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int64_t i = 0; i < input_elems; i++) {
      input[i] = dis(gen);
    }

    moe_sum_ref(topk, r_output.data(), input.data(), num_tokens, hidden_size);

    iris::DMem mem_input(input.data(), input_size_bytes);
    iris::DMem mem_output(output.data(), output_size_bytes);
    iris::DMem mem_output_vec4(output_vec4.data(), output_size_bytes);

    int64_t nano_seconds = run_kernel(scalar_kernel_name(topk), mem_input, mem_output,
                                      hidden_size, num_tokens, repeat, false);
    bool ok = true;
    for (int64_t i = 0; i < output_elems; i++) {
      if (std::fabs(r_output[i] - output[i]) > 1e-4f) {
        ok = false;
        break;
      }
    }
    std::printf("Average execution time of kernel (TopK = %d): %f (us)\n",
                topk, (nano_seconds * 1e-3f) / repeat);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    failures += ok ? 0 : 1;

    float io_bytes = 1.f * repeat * (input_size_bytes + output_size_bytes);
    float bw = io_bytes / nano_seconds;
    std::printf("Kernel bandwidth: %f GB/s \n", bw);

    nano_seconds = run_kernel(vec4_kernel_name(topk), mem_input, mem_output_vec4,
                              hidden_size, num_tokens, repeat, true);
    const int rc = std::memcmp(output.data(), output_vec4.data(), output_size_bytes);
    std::printf("Average execution time of vec4 kernel (TopK = %d): %f (us)\n",
                topk, (nano_seconds * 1e-3f) / repeat);
    std::printf("%s\n", rc ? "FAIL" : "PASS");
    failures += rc ? 1 : 0;

    float bw_vec4 = io_bytes / nano_seconds;
    std::printf("Kernel(vec4) bandwidth: %f GB/s (%f%%)\n", bw_vec4, 100 * (bw_vec4 - bw) / bw);
  }

  platform.finalize();
  return failures == 0 ? 0 : 1;
}
