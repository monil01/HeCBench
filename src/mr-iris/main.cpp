#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

#include "benchmark.h"

static int run_kernel(const char* kernel_name, const uint32_t* bases,
                      const uint32_t* nums, int iter)
{
  int val = 0;
  iris::DMem d_bases(const_cast<uint32_t*>(bases), sizeof(bases32));
  iris::DMem d_nums(const_cast<uint32_t*>(nums), iter * sizeof(uint32_t));
  iris::DMem d_val(&val, sizeof(int));

  size_t gws[1] = {static_cast<size_t>(((iter + 255) / 256) * 256)};
  size_t lws[1] = {256};
  int iter_arg = iter;
  void* params[4] = {&d_bases, &d_nums, &d_val, &iter_arg};
  int pinfo[4] = {iris_r, iris_r, iris_rw, sizeof(iter_arg)};
  iris::Task task;
  task.kernel(kernel_name, 1, nullptr, gws, lws, 4, params, pinfo);
  task.flush_out(d_val);
  task.submit(iris_gpu, nullptr, 1);
  d_val.fetch(&val);
  return val;
}

static void run_benchmark(const int repeat)
{
  int i, j, cnt, val_ref, val_eff;
  uint64_t time_vals[SIZES_CNT_MAX][BASES_CNT_MAX][2];

  printf("Starting benchmark...\n");

  bool ok = true;
  double mr32_sf_time = 0.0, mr32_eff_time = 0.0;

  for (i = 0; i < SIZES_CNT32; i++) {
    val_ref = val_eff = 0;

    for (cnt = 1; cnt <= BASES_CNT32; cnt++) {
      time_point start = get_time();
      for (j = 0; j < BENCHMARK_ITERATIONS; j++)
        val_eff += efficient_mr32(bases32, cnt, n32[i][j]);
      time_vals[i][cnt - 1][0] = elapsed_time(start);
    }

    for (cnt = 1; cnt <= BASES_CNT32; cnt++) {
      time_point start = get_time();
      for (j = 0; j < BENCHMARK_ITERATIONS; j++)
        val_ref += straightforward_mr32(bases32, cnt, n32[i][j]);
      time_vals[i][cnt - 1][1] = elapsed_time(start);
    }

    if (val_ref != val_eff) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_ref = %d, val_eff = %d\n", val_ref, val_eff);
      break;
    }

    double time = 0.0;
    int val_dev = 0;
    for (int n = 0; n < repeat; n++) {
      auto start = std::chrono::steady_clock::now();
      val_dev = run_kernel("mr32_sf", bases32, n32[i], BENCHMARK_ITERATIONS);
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    mr32_sf_time += time / repeat;

    if (val_ref != val_dev) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
      break;
    }

    time = 0.0;
    for (int n = 0; n < repeat; n++) {
      auto start = std::chrono::steady_clock::now();
      val_dev = run_kernel("mr32_eff", bases32, n32[i], BENCHMARK_ITERATIONS);
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    mr32_eff_time += time / repeat;

    if (val_ref != val_dev) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
      break;
    }
  }

  printf("Total kernel execution time (mr32_simple  ): %f (ms)\n", mr32_sf_time * 1e-6);
  printf("Total kernel execution time (mr32_efficent): %f (ms)\n", mr32_eff_time * 1e-6);
  printf("%s\n", ok ? "PASS" : "FAIL");

  print_results(bits32, SIZES_CNT32, BASES_CNT32, time_vals);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = atoi(argv[1]);
  if (repeat <= 0) {
    platform.finalize();
    return 1;
  }

  printf("Setting random primes...\n");
  set_nprimes();
  run_benchmark(repeat);

  printf("Setting random odd integers...\n");
  set_nintegers();
  run_benchmark(repeat);

  platform.finalize();
  return 0;
}
