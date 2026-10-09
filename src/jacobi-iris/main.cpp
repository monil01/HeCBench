#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>

#include "jacobi.h"

static void initialize_data(float* f)
{
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == N - 1) {
        f[IDX(i, j)] = sinf(j * 2 * M_PI / (N - 1));
      } else if (j == 0 || j == N - 1) {
        f[IDX(i, j)] = sinf(i * 2 * M_PI / (N - 1));
      } else {
        f[IDX(i, j)] = 0.0f;
      }
    }
  }
}

static void jacobi_step(iris::DMem& mem_f, iris::DMem& mem_f_old, iris::DMem& mem_error)
{
  const size_t gws[2] = { N, N };
  const size_t lws[2] = { 16, 16 };

  iris::Task task;
  void* params[3] = { &mem_f, &mem_f_old, &mem_error };
  int pinfo[3] = { iris_rw, iris_r, iris_rw };
  task.kernel("jacobi_step", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 3, params, pinfo);
  task.flush_out(mem_error);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  auto start_time = std::chrono::steady_clock::now();

  float* f = static_cast<float*>(aligned_alloc(64, N * N * sizeof(float)));
  float* f_old = static_cast<float*>(aligned_alloc(64, N * N * sizeof(float)));
  if (!f || !f_old) {
    std::cerr << "host allocation failed\n";
    free(f);
    free(f_old);
    platform.finalize();
    return 1;
  }

  initialize_data(f);
  initialize_data(f_old);

  iris::DMem mem_f(f, N * N * sizeof(float));
  iris::DMem mem_f_old(f_old, N * N * sizeof(float));

  float error = std::numeric_limits<float>::max();
  float error_accum = 0.0f;
  iris::DMem mem_error(&error_accum, sizeof(float));

  const float tolerance = 1.e-5f;
  const int max_iters = 10000;
  int num_iters = 0;

  iris::DMem* cur = &mem_f;
  iris::DMem* old = &mem_f_old;

  auto start = std::chrono::steady_clock::now();
  while (error > tolerance && num_iters < max_iters) {
    error_accum = 0.0f;
    mem_error.update(&error_accum);

    jacobi_step(*cur, *old, mem_error);
    std::swap(cur, old);

    mem_error.fetch(&error_accum);
    error = std::sqrt(error_accum / (N * N));

    if (num_iters % 1000 == 0) {
      std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
    }
    ++num_iters;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  int rc = 0;
  if (error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
    rc = 1;
  }

  free(f);
  free(f_old);
  auto end_time = std::chrono::steady_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
  double duration = total_time * 1e-9;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  platform.finalize();
  return rc;
}
