#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX_GPU_COUNT
#define MAX_GPU_COUNT 8
#endif

constexpr int DATA_N = 1048576 * 32;
constexpr int BLOCK_N = 32;
constexpr int THREAD_N = 256;
constexpr int ACCUM_N = BLOCK_N * THREAD_N;

struct GPUPlan {
  int dataN = 0;
  std::vector<float> h_Data;
  std::vector<float> h_Sum_from_device;
  float h_Sum = 0.0f;
};

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = std::atoi(argv[1]);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::printf("Starting simpleMultiDevice\n");
  int gpu_n = MIN(platform.ndevs(), MAX_GPU_COUNT);
  if (gpu_n <= 0) {
    std::fprintf(stderr, "No IRIS HIP devices available\n");
    platform.finalize();
    return 1;
  }

  std::printf("GPU device count: %i\n", gpu_n);
  std::printf("Generating input data of size %d ...\n\n", DATA_N);

  std::vector<GPUPlan> plan(gpu_n);
  for (int i = 0; i < gpu_n; i++) {
    plan[i].dataN = DATA_N / gpu_n;
  }
  for (int i = 0; i < DATA_N % gpu_n; i++) {
    plan[i].dataN++;
  }

  for (int i = 0; i < gpu_n; i++) {
    plan[i].h_Data.resize(plan[i].dataN);
    plan[i].h_Sum_from_device.assign(ACCUM_N, 0.0f);
    for (int j = 0; j < plan[i].dataN; j++) {
      plan[i].h_Data[j] = static_cast<float>(std::rand()) /
                          static_cast<float>(RAND_MAX);
    }
  }

  std::vector<iris::DMem> mem_data;
  std::vector<iris::DMem> mem_sum;
  mem_data.reserve(gpu_n);
  mem_sum.reserve(gpu_n);
  for (int i = 0; i < gpu_n; i++) {
    mem_data.emplace_back(plan[i].h_Data.data(),
                          plan[i].h_Data.size() * sizeof(float));
    mem_sum.emplace_back(plan[i].h_Sum_from_device.data(),
                         plan[i].h_Sum_from_device.size() * sizeof(float));
  }

  std::printf("Computing with %d GPUs...\n", gpu_n);
  auto start = std::chrono::steady_clock::now();

  size_t gws[1] = {static_cast<size_t>(ACCUM_N)};
  size_t lws[1] = {THREAD_N};
  for (int k = 0; k < repeat; k++) {
    for (int i = 0; i < gpu_n; i++) {
      int data_n = plan[i].dataN;
      iris::Task task;
      void* params[3] = {&mem_sum[i], &mem_data[i], &data_n};
      int pinfo[3] = {iris_w, iris_r, sizeof(data_n)};
      task.kernel("reduceKernel", 1, nullptr, gws, lws, 3, params, pinfo);
      if (k == repeat - 1) {
        task.flush_out(mem_sum[i]);
      }
      task.submit(i, nullptr, 1);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("  Average GPU Processing time: %f (us)\n\n",
              time * 1e-3f / repeat);

  float sumGPU = 0.0f;
  for (int i = 0; i < gpu_n; i++) {
    mem_sum[i].fetch(plan[i].h_Sum_from_device.data());
    float sum = 0.0f;
    for (int j = 0; j < ACCUM_N; j++) {
      sum += plan[i].h_Sum_from_device[j];
    }
    plan[i].h_Sum = sum;
    sumGPU += plan[i].h_Sum;
  }

  std::printf("Computing with Host CPU...\n\n");
  double sumCPU = 0.0;
  for (int i = 0; i < gpu_n; i++) {
    for (int j = 0; j < plan[i].dataN; j++) {
      sumCPU += plan[i].h_Data[j];
    }
  }

  std::printf("Comparing GPU and Host CPU results...\n");
  double diff = std::fabs(sumCPU - sumGPU) / std::fabs(sumCPU);
  std::printf("  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
  std::printf("  Relative difference: %E \n\n", diff);

  platform.finalize();
  return (diff < 1e-5) ? EXIT_SUCCESS : EXIT_FAILURE;
}
