#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#define BLOCK_SIZE 256

static void SwapElements(int step, int stage, int num_sequence, int seq_len,
                         int* array)
{
  for (int seq_num = 0; seq_num < num_sequence; seq_num++) {
    int odd = seq_num / static_cast<int>(std::pow(2, (step - stage)));
    bool increasing = ((odd % 2) == 0);
    int h_len = seq_len / 2;

    for (int i = seq_num * seq_len; i < seq_num * seq_len + h_len; i++) {
      int swapped_ele = i + h_len;
      if (((array[i] > array[swapped_ele]) && increasing) ||
          ((array[i] < array[swapped_ele]) && !increasing)) {
        int temp = array[i];
        array[i] = array[swapped_ele];
        array[swapped_ele] = temp;
      }
    }
  }
}

static void BitonicSort(int a[], int n)
{
  for (int step = 0; step < n; step++) {
    for (int stage = step; stage >= 0; stage--) {
      int num_sequence = static_cast<int>(std::pow(2, (n - stage - 1)));
      int sequence_len = static_cast<int>(std::pow(2, stage + 1));
      SwapElements(step, stage, num_sequence, sequence_len, a);
    }
  }
}

static void Usage(const std::string& prog_name, int exponent)
{
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: " << prog_name << " n k \n\n";
  std::cout << " n: Integer exponent presenting the size of the input array. "
               "The number of element in\n";
  std::cout << "    the array must be power of 2 (e.g., 1, 2, 4, ...). Please "
               "enter the corresponding\n";
  std::cout << "    exponent between 0 and " << exponent - 1 << ".\n";
  std::cout << " k: Seed used to generate a random sequence.\n";
}

static void ParallelBitonicSort(int input[], int n)
{
  int size = static_cast<int>(std::pow(2, n));
  size_t size_bytes = sizeof(int) * static_cast<size_t>(size);

  iris::DMem mem_input(input, size_bytes);

  auto start = std::chrono::steady_clock::now();
  for (int step = 0; step < n; step++) {
    for (int stage = step; stage >= 0; stage--) {
      int seq_len = static_cast<int>(std::pow(2, stage + 1));
      int two_power = 1 << (step - stage);
      size_t gws[] = { static_cast<size_t>(size) };
      size_t lws[] = { BLOCK_SIZE };
      void* params[] = { &seq_len, &two_power, &mem_input };
      int pinfo[] = { sizeof(seq_len), sizeof(two_power), iris_rw };

      iris::Task task;
      task.kernel("bitonic_sort", 1, nullptr, gws, lws, 3, params, pinfo);
      if (step == n - 1 && stage == 0) {
        task.flush_out(mem_input);
      }
      task.submit(iris_gpu, nullptr, 1);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Total kernel execution time: %f (ms)\n", time * 1e-6f);
  mem_input.fetch(input);
}

int main(int argc, char* argv[])
{
  int n, seed, size;
  int exp_max = static_cast<int>(std::log2(std::numeric_limits<int>::max()));

  try {
    n = std::stoi(argv[1]);
    if (n < 0 || n >= exp_max) {
      Usage(argv[0], exp_max);
      return -1;
    }
    seed = std::stoi(argv[2]);
    size = static_cast<int>(std::pow(2, n));
  } catch (...) {
    Usage(argv[0], exp_max);
    return -1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::cout << "\nArray size: " << size << ", seed: " << seed << "\n";

  size_t size_bytes = static_cast<size_t>(size) * sizeof(int);
  int* data_cpu = static_cast<int*>(std::malloc(size_bytes));
  int* data_gpu = static_cast<int*>(std::malloc(size_bytes));
  if (!data_cpu || !data_gpu) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(data_cpu);
    std::free(data_gpu);
    platform.finalize();
    return 1;
  }

  std::srand(seed);
  for (int i = 0; i < size; i++) {
    data_gpu[i] = data_cpu[i] = std::rand() % 1000;
  }

  std::cout << "Bitonic sort (parallel)..\n";
  ParallelBitonicSort(data_gpu, n);

  std::cout << "Bitonic sort (serial)..\n";
  BitonicSort(data_cpu, n);

  int unequal = std::memcmp(data_gpu, data_cpu, size_bytes);
  std::cout << (unequal ? "FAIL" : "PASS") << std::endl;

  std::free(data_cpu);
  std::free(data_gpu);
  platform.finalize();
  return unequal ? 1 : 0;
}
