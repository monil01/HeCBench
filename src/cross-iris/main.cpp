#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>
#include <vector>

#define BLOCK_SIZE 256

template <typename T>
struct KernelNames;

template <>
struct KernelNames<float> {
  static constexpr const char* cross1 = "cross_kernel_float";
  static constexpr const char* cross2 = "cross2_kernel_float";
  static constexpr const char* cross3 = "cross3_kernel_float";
};

template <>
struct KernelNames<double> {
  static constexpr const char* cross1 = "cross_kernel_double";
  static constexpr const char* cross2 = "cross2_kernel_double";
  static constexpr const char* cross3 = "cross3_kernel_double";
};

template <typename T>
static void run_cross(
    const char* kernel_name,
    const char* label,
    int nrows,
    int repeat,
    iris::DMem& mem_out,
    iris::DMem& mem_a,
    iris::DMem& mem_b)
{
  const int stride = 1;
  const size_t blocks = (static_cast<size_t>(nrows) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[7] = {
      &nrows,
      &mem_out,
      &mem_a,
      &mem_b,
      const_cast<int*>(&stride),
      const_cast<int*>(&stride),
      const_cast<int*>(&stride)
    };
    int pinfo[7] = {
      sizeof(nrows),
      iris_w,
      iris_r,
      iris_r,
      sizeof(stride),
      sizeof(stride),
      sizeof(stride)
    };
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 7, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of %s kernel: %f (us)\n", label, (time * 1e-3f) / repeat);
}

template <typename T>
static void run_cross3(
    int nrows,
    int repeat,
    iris::DMem& mem_out,
    iris::DMem& mem_a,
    iris::DMem& mem_b)
{
  const size_t blocks = (static_cast<size_t>(nrows) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {
      &nrows,
      &mem_out,
      &mem_a,
      &mem_b
    };
    int pinfo[4] = {
      sizeof(nrows),
      iris_w,
      iris_r,
      iris_r
    };
    task.kernel(KernelNames<T>::cross3, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of cross3 kernel: %f (us)\n", (time * 1e-3f) / repeat);
}

template <typename T>
static void eval(int nrows, int repeat)
{
  const int num_elems = nrows * 3;
  const size_t size_bytes = static_cast<size_t>(num_elems) * sizeof(T);

  std::vector<T> a(num_elems);
  std::vector<T> b(num_elems);
  std::vector<T> o(num_elems);
  std::vector<T> o2(num_elems);
  std::vector<T> o3(num_elems);

  std::default_random_engine g(123);
  std::uniform_real_distribution<T> distr(-2.f, 2.f);
  for (int i = 0; i < num_elems; i++) {
    a[i] = distr(g);
    b[i] = distr(g);
  }

  iris::DMem mem_a(a.data(), size_bytes);
  iris::DMem mem_b(b.data(), size_bytes);
  iris::DMem mem_o(o.data(), size_bytes);

  run_cross<T>(KernelNames<T>::cross1, "cross1", nrows, repeat, mem_o, mem_a, mem_b);
  mem_o.fetch(o.data());

  run_cross<T>(KernelNames<T>::cross2, "cross2", nrows, repeat, mem_o, mem_a, mem_b);
  mem_o.fetch(o2.data());

  run_cross3<T>(nrows, repeat, mem_o, mem_a, mem_b);
  mem_o.fetch(o3.data());

  bool ok = true;
  for (int i = 0; i < num_elems; i++) {
    if (std::fabs(o[i] - o2[i]) > static_cast<T>(1e-3) ||
        std::fabs(o[i] - o3[i]) > static_cast<T>(1e-3)) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of rows in a 2D tensor> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nrows = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (nrows <= 0 || repeat <= 0) {
    std::printf("Usage: %s <number of rows in a 2D tensor> <repeat>\n", argv[0]);
    platform.finalize();
    return 1;
  }

  std::printf("=========== Data type is FP32 ==========\n");
  eval<float>(nrows, repeat);

  std::printf("=========== Data type is FP64 ==========\n");
  eval<double>(nrows, repeat);

  platform.finalize();
  return 0;
}
