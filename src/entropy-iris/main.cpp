#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int width = std::atoi(argv[1]);
  int height = std::atoi(argv[2]);
  int repeat = std::atoi(argv[3]);

  const int input_bytes = width * height * static_cast<int>(sizeof(char));
  const int output_bytes = width * height * static_cast<int>(sizeof(float));
  std::vector<char> input(static_cast<size_t>(width) * height);
  std::vector<float> output(static_cast<size_t>(width) * height);
  std::vector<float> output_ref(static_cast<size_t>(width) * height);
  std::vector<float> log_table(26);

  std::srand(123);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      input[static_cast<size_t>(i) * width + j] = std::rand() % 16;
    }
  }

  iris::DMem d_input(input.data(), input_bytes);
  iris::DMem d_output(output.data(), output_bytes);

  const size_t gws[2] = {
      static_cast<size_t>(((width + 15) / 16) * 16),
      static_cast<size_t>(((height + 15) / 16) * 16)};
  const size_t lws[2] = {16, 16};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {&d_output, &d_input, &height, &width};
    int pinfo[4] = {iris_w, iris_r, sizeof(height), sizeof(width)};
    task.kernel("entropy", 2, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 4, params, pinfo);
    task.flush_out(d_output);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel (baseline) execution time %f (s)\n",
              (time * 1e-9f) / repeat);

  for (int i = 0; i <= 25; i++) {
    log_table[static_cast<size_t>(i)] = i <= 1 ? 0.0f : i * std::log2(static_cast<float>(i));
  }
  iris::DMem d_log_table(log_table.data(), sizeof(float) * log_table.size());

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[5] = {&d_output, &d_input, &d_log_table, &height, &width};
    int pinfo[5] = {iris_w, iris_r, iris_r, sizeof(height), sizeof(width)};
    task.kernel("entropy_opt_16_16", 2, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 5, params, pinfo);
    task.flush_out(d_output);
    task.submit(iris_gpu, nullptr, 1);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel (optimized) execution time %f (s)\n",
              (time * 1e-9f) / repeat);

  d_output.fetch(output.data());

  reference(output_ref.data(), input.data(), height, width);

  bool ok = true;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      size_t idx = static_cast<size_t>(i) * width + j;
      if (std::fabs(output[idx] - output_ref[idx]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    if (!ok) break;
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  return ok ? 0 : 1;
}
