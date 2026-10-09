#include <iris/iris.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define TREE_NUM 4096
#define TREE_SIZE 4096
#define GROUP_SIZE 256

bool Verify(const int* deviceResult, const int* reference, int treeNumber) {
  for (int i = 0; i < treeNumber; i++) {
    if (deviceResult[i] != reference[i]) return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int iterations = std::atoi(argv[1]);
  const int treeSize = TREE_SIZE;
  const int treeNumber = TREE_NUM;
  bool fail = false;

  if (iterations < 1) {
    std::cout << "Iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }

  if (treeNumber < GROUP_SIZE) {
    std::cout << "treeNumber should be larger than the work group size"
              << std::endl;
    return -1;
  }
  if (treeNumber % 256 != 0) {
    std::cout << "treeNumber should be a multiple of 256" << std::endl;
    return -1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int elements = treeSize * treeNumber;
  size_t inputSize = elements * sizeof(int);
  size_t outputSize = treeNumber * sizeof(int);

  int* data = static_cast<int*>(std::malloc(inputSize));
  int* deviceResult = static_cast<int*>(std::malloc(outputSize));
  int* reference = static_cast<int*>(std::malloc(outputSize));
  std::memset(reference, 0, outputSize);
  for (int i = 0; i < treeNumber; i++)
    for (int j = 0; j < treeSize; j++)
      reference[i] += i * treeSize + j;

  size_t gws = treeNumber;
  size_t lws = GROUP_SIZE;
  int n = treeNumber;

  for (int i = 0; i < treeNumber; i++)
    for (int j = 0; j < treeSize; j++)
      data[j + i * treeSize] = j + i * treeSize;

  {
    iris::DMem mem_in(data, inputSize);
    iris::DMem mem_out(deviceResult, outputSize);
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
      iris::Task task;
      void* params[3] = {&mem_in, &mem_out, &n};
      int pinfo[3] = {iris_r, iris_w, sizeof(n)};
      task.kernel("AoSKernel", 1, nullptr, &gws, &lws, 3, params, pinfo);
      if (iter == iterations - 1) task.flush_out(mem_out);
      task.submit(iris_gpu, nullptr, 1);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                     start)
                    .count();
    std::cout << "Average kernel execution time (AoS): "
              << (time * 1e-3f) / iterations << " (us)\n";
  }

  if (!Verify(deviceResult, reference, treeNumber)) fail = true;

  if (fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";

  for (int i = 0; i < treeNumber; i++)
    for (int j = 0; j < treeSize; j++)
      data[i + j * treeNumber] = j + i * treeSize;

  {
    iris::DMem mem_in(data, inputSize);
    iris::DMem mem_out(deviceResult, outputSize);
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
      iris::Task task;
      void* params[3] = {&mem_in, &mem_out, &n};
      int pinfo[3] = {iris_r, iris_w, sizeof(n)};
      task.kernel("SoAKernel", 1, nullptr, &gws, &lws, 3, params, pinfo);
      if (iter == iterations - 1) task.flush_out(mem_out);
      task.submit(iris_gpu, nullptr, 1);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                     start)
                    .count();
    std::cout << "Average kernel execution time (SoA): "
              << (time * 1e-3f) / iterations << " (us)\n";
  }

  if (!Verify(deviceResult, reference, treeNumber)) fail = true;

  if (fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";

  std::free(deviceResult);
  std::free(reference);
  std::free(data);
  platform.finalize();
  return 0;
}
