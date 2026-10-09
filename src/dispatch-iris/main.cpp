#include <iris/iris.hpp>
#include <iris/iris_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <string>

#define NUM_GROUPS 1
#define GROUP_SIZE 1
#define WARMUP_RUN_COUNT 100
#define TIMING_RUN_COUNT 1000
#define TOTAL_RUN_COUNT WARMUP_RUN_COUNT + TIMING_RUN_COUNT
#define BATCH_SIZE 1000

static void print_timing(const std::string& test,
                         const std::array<float, TOTAL_RUN_COUNT>& results,
                         int batch = 1) {
  float total_us = 0.0f;
  float mean_us = 0.0f;
  float stddev_us = 0.0f;

  auto start_iter = std::next(results.begin(), WARMUP_RUN_COUNT);
  auto end_iter = results.end();

  std::for_each(start_iter, end_iter, [&](const float& run_ms) {
    total_us += (run_ms * 1000) / batch;
  });
  mean_us = total_us / TIMING_RUN_COUNT;

  total_us = 0;
  std::for_each(start_iter, end_iter, [&](const float& run_ms) {
    float dev_us = ((run_ms * 1000) / batch) - mean_us;
    total_us += dev_us * dev_us;
  });
  stddev_us = std::sqrt(total_us / TIMING_RUN_COUNT);

  std::printf("\n %s: mean = %.1f us, stddev = %.1f us\n", test.c_str(),
              mean_us, stddev_us);
}

static void submit_empty(int sync) {
  iris::Task task;
  size_t gws[] = {NUM_GROUPS * GROUP_SIZE};
  size_t lws[] = {GROUP_SIZE};
  task.kernel("EmptyKernel", 1, nullptr, gws, lws, 0, nullptr, nullptr);
  task.submit(iris_gpu, nullptr, sync);
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::array<float, TOTAL_RUN_COUNT> results;

  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    submit_empty(0);
    auto stop = std::chrono::high_resolution_clock::now();
    results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
  }
  iris_synchronize();
  print_timing("Enqueue rate", results);

  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    submit_empty(1);
    auto stop = std::chrono::high_resolution_clock::now();
    results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
  }
  print_timing("Single dispatch latency", results);

  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < BATCH_SIZE; j++) submit_empty(0);
    iris_synchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
  }
  print_timing("Batch dispatch latency", results, BATCH_SIZE);

  platform.finalize();
  return 0;
}
