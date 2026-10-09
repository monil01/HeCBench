#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

template <typename T>
struct KernelName;

template <>
struct KernelName<unsigned char> {
  static constexpr const char* value = "sortkv_u8";
};

template <>
struct KernelName<short> {
  static constexpr const char* value = "sortkv_i16";
};

template <>
struct KernelName<int> {
  static constexpr const char* value = "sortkv_i32";
};

template <>
struct KernelName<long> {
  static constexpr const char* value = "sortkv_i64";
};

template <typename T>
static void run_sort_task(std::vector<int>& keys, std::vector<T>& vals) {
  const int n = static_cast<int>(keys.size());
  iris::DMem mem_keys(keys.data(), keys.size() * sizeof(int));
  iris::DMem mem_vals(vals.data(), vals.size() * sizeof(T));

  const size_t block = 256;
  const size_t gws0 = ((static_cast<size_t>(n) + block - 1) / block) * block;
  size_t gws[] = {gws0};
  size_t lws[] = {block};
  void* params[] = {&mem_keys, &mem_vals, const_cast<int*>(&n)};
  int pinfo[] = {iris_rw, iris_rw, sizeof(n)};

  iris::Task task;
  task.kernel(KernelName<T>::value, 1, nullptr, gws, lws, 3, params, pinfo);
  task.flush_out(mem_keys);
  task.flush_out(mem_vals);
  task.submit(iris_gpu, nullptr, 1);
}

template <typename T>
static void sort_key_value(int n, int repeat, bool verify) {
  std::printf("Number of keys is %d and the size of each value in bytes is %zu\n",
              n, sizeof(T));

  const unsigned seed = 123;
  bool ok = true;

  std::vector<int> keys(n);
  std::vector<T> vals(n);
  std::iota(keys.begin(), keys.end(), 0);

  double total_time = 0.0;
  for (int iter = 0; iter < repeat; iter++) {
    std::mt19937 engine(seed);
    std::shuffle(keys.begin(), keys.end(), engine);
    for (int i = 0; i < n; i++) vals[i] = static_cast<T>(keys[i] % 256);

    auto start = std::chrono::steady_clock::now();
    run_sort_task(keys, vals);
    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    total_time += time;
  }

  if (!verify) {
    std::printf("Average sort time %f (us)\n", (total_time * 1e-3) / repeat);
  } else {
    for (int i = 0; i < n; i++) {
      if (keys[i] != i || vals[i] != static_cast<T>(i % 256)) {
        ok = false;
        break;
      }
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
  }
}

int main(int argc, char* argv[]) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  if (argc != 3) {
    std::printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    platform.finalize();
    return 1;
  }
  const int size = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (size <= 0 || repeat <= 0) {
    std::printf("size and repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  std::printf("\nWarmup and verify\n");
  sort_key_value<unsigned char>(size, repeat, true);
  sort_key_value<short>(size, repeat, true);
  sort_key_value<int>(size, repeat, true);
  sort_key_value<long>(size, repeat, true);

  std::printf("\nPerformance evaluation\n");
  sort_key_value<unsigned char>(size, repeat, false);
  sort_key_value<short>(size, repeat, false);
  sort_key_value<int>(size, repeat, false);
  sort_key_value<long>(size, repeat, false);

  platform.finalize();
  return 0;
}
