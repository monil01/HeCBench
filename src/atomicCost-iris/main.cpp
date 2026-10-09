#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define BLOCK_SIZE 256

template <typename T>
static int atomicCost(const char* with_atomic_kernel,
                      const char* without_atomic_kernel,
                      int length,
                      int size,
                      int repeat)
{
  std::printf("\n\n");
  std::printf("Each thread sums up %d elements\n", size);

  const int num_threads = length / size;
  assert(length % size == 0);
  assert(num_threads % BLOCK_SIZE == 0);

  const size_t result_size = sizeof(T) * static_cast<size_t>(num_threads);
  T* result_wi = static_cast<T*>(std::calloc(static_cast<size_t>(num_threads), sizeof(T)));
  T* result_wo = static_cast<T*>(std::calloc(static_cast<size_t>(num_threads), sizeof(T)));
  if (!result_wi || !result_wo) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(result_wi);
    std::free(result_wo);
    return 1;
  }

  iris::DMem mem_wi(result_wi, result_size);
  iris::DMem mem_wo(result_wo, result_size);

  const size_t gws[1] = {static_cast<size_t>(num_threads)};
  const size_t lws[1] = {BLOCK_SIZE};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = {&mem_wi, &size};
    int pinfo[2] = {iris_rw, sizeof(size)};
    task.kernel(with_atomic_kernel, 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 2, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  iris::Task flush_wi;
  flush_wi.flush_out(mem_wi);
  flush_wi.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
              time * 1e-3f / repeat);
  mem_wi.fetch(result_wi);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = {&mem_wo, &size};
    int pinfo[2] = {iris_rw, sizeof(size)};
    task.kernel(without_atomic_kernel, 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 2, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  iris::Task flush_wo;
  flush_wo.flush_out(mem_wo);
  flush_wo.submit(iris_gpu, nullptr, 1);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
              time * 1e-3f / repeat);
  mem_wo.fetch(result_wo);

  const int diff = std::memcmp(result_wi, result_wo, result_size);
  std::printf("%s\n", diff ? "FAIL" : "PASS");

  std::free(result_wi);
  std::free(result_wo);
  return diff ? 1 : 0;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <N> <repeat>\n", argv[0]);
    std::printf("N: the number of elements to sum per thread (1 - 16)\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nelems = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  const int length = 922521600;
  assert(length % BLOCK_SIZE == 0);

  int errors = 0;

  std::printf("\nFP64 atomic add\n");
  errors += atomicCost<double>("wiAtomicOnGlobalMem_f64", "woAtomicOnGlobalMem_f64",
                               length, nelems, repeat);

  std::printf("\nINT32 atomic add\n");
  errors += atomicCost<int>("wiAtomicOnGlobalMem_i32", "woAtomicOnGlobalMem_i32",
                            length, nelems, repeat);

  std::printf("\nFP32 atomic add\n");
  errors += atomicCost<float>("wiAtomicOnGlobalMem_f32", "woAtomicOnGlobalMem_f32",
                              length, nelems, repeat);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}
