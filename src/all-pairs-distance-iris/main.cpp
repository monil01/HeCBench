#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define INSTANCES 224
#define ATTRIBUTES 4096
#define THREADS 128

static void CPU(int* data, int* distance)
{
#pragma omp parallel for collapse(2)
  for (int i = 0; i < INSTANCES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      for (int k = 0; k < ATTRIBUTES; k++) {
        distance[i + INSTANCES * j] +=
          (data[i * ATTRIBUTES + k] != data[j * ATTRIBUTES + k]);
      }
    }
  }
}

static bool run_kernel(const char* kernel_name, int iterations, char* data_char,
                       int* cpu_distance, int* gpu_distance)
{
  iris::DMem mem_data(data_char, INSTANCES * ATTRIBUTES * sizeof(char));
  iris::DMem mem_distance(gpu_distance, INSTANCES * INSTANCES * sizeof(int));

  const size_t gws[2] = {INSTANCES * THREADS, INSTANCES};
  const size_t lws[2] = {THREADS, 1};
  const int total = INSTANCES * INSTANCES;

  const auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < iterations; n++) {
    std::memset(gpu_distance, 0, INSTANCES * INSTANCES * sizeof(int));
    iris::Task zero_task;
    void* zero_params[2] = {&mem_distance, const_cast<int*>(&total)};
    int zero_pinfo[2] = {iris_w, sizeof(total)};
    const size_t zero_gws[1] = {static_cast<size_t>(((total + 255) / 256) * 256)};
    const size_t zero_lws[1] = {256};
    zero_task.kernel("zero_distance", 1, nullptr, const_cast<size_t*>(zero_gws),
                     const_cast<size_t*>(zero_lws), 2, zero_params, zero_pinfo);
    zero_task.submit(iris_gpu, nullptr, 1);

    iris::Task task;
    void* params[2] = {&mem_data, &mem_distance};
    int pinfo[2] = {iris_r, iris_w};
    task.kernel(kernel_name, 2, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 2, params, pinfo);
    task.flush_out(mem_distance);
    task.submit(iris_gpu, nullptr, 1);
  }
  const auto end = std::chrono::steady_clock::now();
  const double elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::printf("Average kernel execution time: %f (us)\n", elapsed_us / iterations);
  const int status = std::memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  std::printf("%s\n", status ? "FAIL" : "PASS");
  return status == 0;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int iterations = std::atoi(argv[1]);
  int* data = static_cast<int*>(std::malloc(INSTANCES * ATTRIBUTES * sizeof(int)));
  char* data_char = static_cast<char*>(std::malloc(INSTANCES * ATTRIBUTES * sizeof(char)));
  int* cpu_distance = static_cast<int*>(std::malloc(INSTANCES * INSTANCES * sizeof(int)));
  int* gpu_distance = static_cast<int*>(std::malloc(INSTANCES * INSTANCES * sizeof(int)));
  if (!data || !data_char || !cpu_distance || !gpu_distance) {
    std::fprintf(stderr, "allocation failure\n");
    platform.finalize();
    return 1;
  }

  srand(2);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < ATTRIBUTES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      data[i + ATTRIBUTES * j] = data_char[i + ATTRIBUTES * j] = random() % 3;
    }
  }

  auto start = std::chrono::steady_clock::now();
  std::memset(cpu_distance, 0, INSTANCES * INSTANCES * sizeof(int));
  CPU(data, cpu_distance);
  auto end = std::chrono::steady_clock::now();
  double elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::printf("CPU time: %f (us)\n", elapsed_us);

  bool ok = true;
  ok &= run_kernel("k1", iterations, data_char, cpu_distance, gpu_distance);
  ok &= run_kernel("k2", iterations, data_char, cpu_distance, gpu_distance);
  ok &= run_kernel("k3", iterations, data_char, cpu_distance, gpu_distance);

  std::free(cpu_distance);
  std::free(gpu_distance);
  std::free(data);
  std::free(data_char);
  platform.finalize();
  return ok ? 0 : 1;
}
