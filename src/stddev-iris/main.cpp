#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

static void run_stddev(
    iris::DMem& mem_std,
    iris::DMem& mem_data,
    std::vector<float>& zero,
    int D,
    int N,
    bool sample,
    bool flush)
{
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;

  const int grid_x = (N + RowsPerBlk - 1) / RowsPerBlk;
  const int grid_y = (D + ColsPerBlk - 1) / ColsPerBlk;
  const size_t sop_gws[2] = {
    static_cast<size_t>(grid_x) * TPB,
    static_cast<size_t>(grid_y)
  };
  const size_t sop_lws[2] = { TPB, 1 };

  mem_std.update(zero.data());

  iris::Task sop;
  void* sop_params[4] = { &mem_std, &mem_data, &D, &N };
  int sop_pinfo[4] = { iris_rw, iris_r, static_cast<int>(sizeof(D)), static_cast<int>(sizeof(N)) };
  sop.kernel("sopKernel_float_int", 2, nullptr, const_cast<size_t*>(sop_gws),
             const_cast<size_t*>(sop_lws), 4, sop_params, sop_pinfo);
  sop.submit(iris_gpu, nullptr, 1);

  const int sample_size = sample ? N - 1 : N;
  const size_t sample_gws[1] = { static_cast<size_t>(((D + TPB - 1) / TPB) * TPB) };
  const size_t sample_lws[1] = { TPB };
  iris::Task sample_task;
  void* sample_params[3] = { &mem_std, &D, const_cast<int*>(&sample_size) };
  int sample_pinfo[3] = { iris_rw, static_cast<int>(sizeof(D)), static_cast<int>(sizeof(sample_size)) };
  sample_task.kernel("sampleKernel_float_int", 1, nullptr, const_cast<size_t*>(sample_gws),
                     const_cast<size_t*>(sample_lws), 3, sample_params, sample_pinfo);
  if (flush) sample_task.flush_out(mem_std);
  sample_task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
    std::printf("D: number of columns of data (must be a multiple of 32)\n");
    std::printf("N: number of rows of data (at least one row)\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int D = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  const bool sample = true;

  const long input_size = static_cast<long>(D) * N;
  const size_t input_size_bytes = static_cast<size_t>(input_size) * sizeof(float);
  std::vector<float> data(input_size);

  std::srand(123);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      data[static_cast<size_t>(i) * D + j] = std::rand() / static_cast<float>(RAND_MAX);
    }
  }

  std::vector<float> stddev(D, 0.0f);
  std::vector<float> std_ref(D, 0.0f);
  std::vector<float> zero(D, 0.0f);

  iris::DMem mem_data(data.data(), input_size_bytes);
  iris::DMem mem_std(stddev.data(), static_cast<size_t>(D) * sizeof(float));

  run_stddev(mem_std, mem_data, zero, D, N, sample, false);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_stddev(mem_std, mem_data, zero, D, N, sample, i == repeat - 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

  mem_std.fetch(stddev.data());
  stddev_ref(std_ref.data(), data.data(), D, N, sample);

  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (std::fabs(std_ref[i] - stddev[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  platform.finalize();
  return ok ? 0 : 1;
}
