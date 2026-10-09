#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256

static void launch_kernel(const char* name, iris::DMem& mem_a, iris::DMem& mem_o,
                          int n, bool flush)
{
  const size_t gws[1] = {
      static_cast<size_t>(((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)};
  const size_t lws[1] = {BLOCK_SIZE};

  iris::Task task;
  void* params[3] = {&mem_a, &mem_o, &n};
  int pinfo[3] = {iris_r, iris_w, sizeof(n)};
  task.kernel(name, 1, nullptr, const_cast<size_t*>(gws),
              const_cast<size_t*>(lws), 3, params, pinfo);
  if (flush) task.flush_out(mem_o);
  task.submit(iris_gpu, nullptr, 1);
}

static double time_kernel(const char* name, iris::DMem& mem_a, iris::DMem& mem_o,
                          int n, int repeat)
{
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_kernel(name, mem_a, mem_o, n, i == repeat - 1);
  }
  iris_synchronize();
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-9) / repeat;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage %s <n> <repeat>\n", argv[0]);
    return 1;
  }

  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  const size_t size = sizeof(float) * n;

  float* a = static_cast<float*>(std::malloc(size));
  float* o = static_cast<float*>(std::malloc(size));
  float* o0 = static_cast<float*>(std::malloc(size));
  float* o1 = static_cast<float*>(std::malloc(size));
  float* o2 = static_cast<float*>(std::malloc(size));
  if (!a || !o || !o0 || !o1 || !o2) {
    std::fprintf(stderr, "Host allocation failed\n");
    std::free(a);
    std::free(o);
    std::free(o0);
    std::free(o1);
    std::free(o2);
    return 1;
  }

  for (int i = 0; i < n; i++) {
    a[i] = -1.8f + i * (1.79999f / n);
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem mem_a(a, size);
  iris::DMem mem_o0(o0, size);
  iris::DMem mem_o1(o1, size);
  iris::DMem mem_o2(o2, size);
  mem_a.update(a);

  double t0 = time_kernel("k0", mem_a, mem_o0, n, repeat);
  std::printf("Average execution time of k0: %f (s)\n", t0);
  double t1 = time_kernel("k1", mem_a, mem_o1, n, repeat);
  std::printf("Average execution time of k1: %f (s)\n", t1);
  double t2 = time_kernel("k2", mem_a, mem_o2, n, repeat);
  std::printf("Average execution time of k2: %f (s)\n", t2);

  for (int i = 0; i < n; i++) {
    float x = a[i];
    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;
    o[i] = x * (1.f / 3.f - 1.f / 45.f * x2 + 2.f / 945.f * x4 -
                1.f / 4725.f * x6);
  }

  float e[3] = {0, 0, 0};
  for (int i = 0; i < n; i++) {
    e[0] += (o[i] - o0[i]) * (o[i] - o0[i]);
    e[1] += (o[i] - o1[i]) * (o[i] - o1[i]);
    e[2] += (o[i] - o2[i]) * (o[i] - o2[i]);
  }

  std::printf("\nError statistics for the kernels:\n");
  for (int i = 0; i < 3; i++) {
    std::printf("%f ", std::sqrt(e[i]));
  }
  std::printf("\n");

  platform.finalize();
  std::free(a);
  std::free(o);
  std::free(o0);
  std::free(o1);
  std::free(o2);
  return 0;
}
