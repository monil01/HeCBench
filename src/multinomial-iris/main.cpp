#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "reference.h"

#define GPU_NUM_THREADS 256

static void launch_sample_multinomial(
    int* result,
    int num_dist,
    int num_categories,
    float* sample,
    float* distr,
    iris::DMem& mem_result,
    iris::DMem& mem_sample,
    iris::DMem& mem_distr)
{
  (void)result;
  (void)sample;
  (void)distr;

  int stride_dist = num_categories;
  int stride_categories = 1;
  const size_t gws[1] = { 512u * GPU_NUM_THREADS };
  const size_t lws[1] = { GPU_NUM_THREADS };

  iris::Task task;
  void* params[7] = {
    &mem_result,
    &num_dist,
    &num_categories,
    &mem_sample,
    &mem_distr,
    &stride_dist,
    &stride_categories
  };
  int pinfo[7] = {
    iris_w,
    sizeof(num_dist),
    sizeof(num_categories),
    iris_r,
    iris_r,
    sizeof(stride_dist),
    sizeof(stride_categories)
  };
  task.kernel("sampleMultinomialOnce", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 7, params, pinfo);
  task.flush_out(mem_result);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of distributions> <number of categories> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int num_dist = std::atoi(argv[1]);
  const int num_categories = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  const size_t sample_size_bytes = static_cast<size_t>(num_dist) * sizeof(float);
  float* sample = static_cast<float*>(std::malloc(sample_size_bytes));

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> uniform_distr(0.f, 1.f);

  for (int i = 0; i < num_dist; i++) {
    sample[i] = uniform_distr(g);
  }

  const size_t result_size_bytes = static_cast<size_t>(num_dist) * sizeof(int);
  int* result = static_cast<int*>(std::malloc(result_size_bytes));
  int* result_ref = static_cast<int*>(std::malloc(result_size_bytes));

  const size_t distr_size_bytes = static_cast<size_t>(num_dist) * static_cast<size_t>(num_categories) * sizeof(float);
  float* distr = static_cast<float*>(std::malloc(distr_size_bytes));

  if (!sample || !result || !result_ref || !distr) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(result_ref);
    std::free(result);
    std::free(sample);
    std::free(distr);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (int i = 0; i < num_dist; i++) {
    for (int j = 0; j < num_categories; j++) {
      distr[i * num_categories + j] = static_cast<float>(std::rand() % 100 + 1);
    }
  }

  iris::DMem mem_sample(sample, sample_size_bytes);
  iris::DMem mem_distr(distr, distr_size_bytes);
  iris::DMem mem_result(result, result_size_bytes);

  launch_sample_multinomial(result, num_dist, num_categories, sample, distr, mem_result, mem_sample, mem_distr);

  sampleMultinomialOnce_cpu<float, float>(
      result_ref, num_dist, num_categories, sample, distr, num_categories, 1);

  mem_result.fetch(result);

  int error = 0;
  for (int i = 0; i < num_dist; i++) {
    if (std::abs(result[i] - result_ref[i]) > 1) {
      std::printf("results mismatch: %d %d %d\n", i, result[i], result_ref[i]);
      error = 1;
      break;
    }
  }
  std::printf("%s\n", error ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_sample_multinomial(result, num_dist, num_categories, sample, distr, mem_result, mem_sample, mem_distr);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of sampleMultinomialOnce kernel: %f (us)\n", (time * 1e-3f) / repeat);

  std::free(result_ref);
  std::free(result);
  std::free(sample);
  std::free(distr);

  platform.finalize();
  return error;
}
