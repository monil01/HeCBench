#include <iris/iris.hpp>

#include "reference.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

template <class T>
void testcase(const char* kernel_name, const int num, const int repeat)
{
  const size_t len = 1UL << num;
  const unsigned int numThreads = 256;
  const unsigned int numBlocks = (len + numThreads - 1) / numThreads;
  T gpuData[] = {0, 0, (T)-256, 256, 255, 0, 255, 0, 0};
  const T initialData[] = {0, 0, (T)-256, 256, 255, 0, 255, 0, 0};
  const size_t memSize = sizeof(gpuData);

  iris::DMem mem_out(gpuData, memSize);
  size_t gws[1] = {static_cast<size_t>(numBlocks) * numThreads};
  size_t lws[1] = {numThreads};

  for (int i = 0; i < repeat; i++) {
    for (size_t j = 0; j < sizeof(gpuData) / sizeof(gpuData[0]); j++) {
      gpuData[j] = initialData[j];
    }
    mem_out.update(gpuData);

    iris::Task task;
    void* params[2] = {&mem_out, const_cast<size_t*>(&len)};
    int pinfo[2] = {iris_rw, sizeof(len)};
    task.kernel(kernel_name, 1, nullptr, gws, lws, 2, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }

  mem_out.fetch(gpuData);
  computeGold<T>(gpuData, numThreads * numBlocks);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = {&mem_out, const_cast<size_t*>(&len)};
    int pinfo[2] = {iris_rw, sizeof(len)};
    task.kernel(kernel_name, 1, nullptr, gws, lws, 2, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <number of atomic operations> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int num = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  testcase<int>("testKernel_int", num, repeat);
  testcase<unsigned int>("testKernel_uint", num, repeat);

  platform.finalize();
  return 0;
}
