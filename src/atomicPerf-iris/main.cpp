#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define BLOCK_SIZE 256

#include "../atomicPerf-cuda/reference.h"

template <typename T>
const char* suffix();

template <>
const char* suffix<double>() { return "double"; }

template <>
const char* suffix<int>() { return "int"; }

template <>
const char* suffix<float>() { return "float"; }

template <typename T>
bool run_kernel(const char* display_name, const char* symbol_prefix, int n,
                int repeat, const std::vector<T>& input, bool expect_unchanged,
                int fixed_offset) {
  std::vector<T> data = input;
  std::vector<T> ref = input;
  iris::DMem mem(data.data(), data.size() * sizeof(T));
  const size_t gws[] = {static_cast<size_t>(n)};
  const size_t lws[] = {BLOCK_SIZE};
  const std::string kernel_name =
      std::string(symbol_prefix) + "_" + suffix<T>();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    const int offset = fixed_offset >= 0 ? fixed_offset : i % BLOCK_SIZE;
    iris::Task task;
    void* params[] = {&mem, const_cast<int*>(&offset), const_cast<int*>(&n)};
    int pinfo[] = {iris_rw, sizeof(offset), sizeof(n)};
    task.kernel(kernel_name.c_str(), 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 3, params, pinfo);
    task.flush_out(mem);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();

  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of %s: %f (us)\n", display_name,
              elapsed * 1e-3f / repeat);

  if (!expect_unchanged) {
    for (int i = 0; i < repeat; ++i) {
      if (std::strcmp(symbol_prefix, "BlockRangeAtomicOnGlobalMem") == 0) {
        BlockRangeAtomicOnGlobalMem_ref<T>(ref.data(), n);
      } else if (std::strcmp(symbol_prefix, "WarpRangeAtomicOnGlobalMem") ==
                 0) {
        WarpRangeAtomicOnGlobalMem_ref<T>(ref.data(), n);
      } else {
        SingleRangeAtomicOnGlobalMem_ref<T>(ref.data(), i % BLOCK_SIZE, n);
      }
    }
  }

  const bool fail =
      std::memcmp(data.data(), ref.data(), data.size() * sizeof(T)) != 0;
  std::printf("%s\n", fail ? "FAIL" : "PASS");
  return !fail;
}

template <typename T>
void atomicPerf(int n, int len, int repeat) {
  std::vector<T> input(len);
  for (int i = 0; i < len; ++i) input[i] = static_cast<T>(i % 1024 + 1);

  run_kernel<T>("BlockRangeAtomicOnGlobalMem", "BlockRangeAtomicOnGlobalMem",
                n, repeat, input, false, 0);
  run_kernel<T>("WarpRangeAtomicOnGlobalMem", "WarpRangeAtomicOnGlobalMem", n,
                repeat, input, false, 0);
  run_kernel<T>("SingleRangeAtomicOnGlobalMem", "SingleRangeAtomicOnGlobalMem",
                n, repeat, input, false, -1);
  run_kernel<T>("BlockRangeAtomicOnSharedMem", "BlockRangeAtomicOnSharedMem",
                n, repeat, input, true, 0);
  run_kernel<T>("WarpRangeAtomicOnSharedMem", "WarpRangeAtomicOnSharedMem", n,
                repeat, input, true, 0);
  run_kernel<T>("SingleRangeAtomicOnSharedMem", "SingleRangeAtomicOnSharedMem",
                n, repeat, input, true, -1);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int n = 3 * 4 * 7 * 8 * 9 * 256;
  const int len = 1024;

  std::printf("\nFP64 atomic add\n");
  atomicPerf<double>(n, len, repeat);

  std::printf("\nINT32 atomic add\n");
  atomicPerf<int>(n, len, repeat);

  std::printf("\nFP32 atomic add\n");
  atomicPerf<float>(n, len, repeat);

  platform.finalize();
  return 0;
}
