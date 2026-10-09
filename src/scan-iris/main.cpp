#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

template <typename T>
static void verify(const T* cpu_out, const T* gpu_out, int64_t n)
{
  int error = std::memcmp(cpu_out, gpu_out, static_cast<size_t>(n) * sizeof(T));
  if (error) {
    for (int64_t i = 0; i < n; i++) {
      if (cpu_out[i] != gpu_out[i]) {
        std::printf("@%ld: %lf != %lf\n", static_cast<long>(i),
                    static_cast<double>(cpu_out[i]), static_cast<double>(gpu_out[i]));
        break;
      }
    }
  }
  std::printf("%s\n", error ? "FAIL" : "PASS");
}

template <typename T>
static const char* type_name()
{
  if constexpr (std::is_same<T, char>::value) return "char";
  if constexpr (std::is_same<T, short>::value) return "short";
  if constexpr (std::is_same<T, int>::value) return "int";
  return "long";
}

template <typename T, int N>
static void run_test(iris::Platform&, const int64_t n, const int repeat, bool timing)
{
  const int64_t num_blocks = (n + N - 1) / N;
  const int64_t nelems = num_blocks * N;
  const size_t bytes = static_cast<size_t>(nelems) * sizeof(T);
  T* in = static_cast<T*>(std::malloc(bytes));
  T* cpu_out = static_cast<T*>(std::malloc(bytes));
  T* gpu_out = static_cast<T*>(std::malloc(bytes));
  if (!in || !cpu_out || !gpu_out) {
    std::fprintf(stderr, "host allocation failed\n");
    std::exit(1);
  }

  std::srand(123);
  for (int64_t i = 0; i < nelems; i++) in[i] = static_cast<T>(std::rand() % 5 + 1);
  T* t_in = in;
  T* t_out = cpu_out;
  for (int64_t b = 0; b < num_blocks; b++) {
    t_out[0] = 0;
    for (int i = 1; i < N; i++) t_out[i] = t_out[i - 1] + t_in[i - 1];
    t_out += N;
    t_in += N;
  }

  iris::DMem mem_in(in, bytes);
  iris::DMem mem_out(gpu_out, bytes);
  const size_t gws[1] = {1024};
  const size_t lws[1] = {N / 2};
  const std::string base = std::string("scan_") + type_name<T>() + "_" + std::to_string(N);
  const std::string bcao = std::string("scan_bcao_") + type_name<T>() + "_" + std::to_string(N);

  auto launch = [&](const std::string& kernel_name, const char* label) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      iris::Task task;
      void* params[3] = {const_cast<int64_t*>(&num_blocks), &mem_out, &mem_in};
      int pinfo[3] = {sizeof(num_blocks), iris_w, iris_r};
      task.kernel(kernel_name.c_str(), 1, nullptr, const_cast<size_t*>(gws),
                  const_cast<size_t*>(lws), 3, params, pinfo);
      if (i == repeat - 1) task.flush_out(mem_out);
      task.submit(iris_gpu, nullptr, 1);
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (timing) {
      std::printf("Element size in bytes is %zu. Average execution time of %s: %f (us)\n",
                  sizeof(T), label, (elapsed * 1e-3f) / repeat);
    } else {
      verify(cpu_out, gpu_out, nelems);
    }
  };

  launch(base, "scan (w/  bank conflicts)");
  launch(bcao, "scan (w/o bank conflicts)");
  std::free(gpu_out);
  std::free(cpu_out);
  std::free(in);
}

template <int N>
static void run(iris::Platform& platform, const int64_t n, const int repeat)
{
  for (int i = 0; i < 2; i++) {
    bool report_timing = i > 0;
    std::printf("\nThe number of elements to scan in a thread block: %d\n", N);
    run_test<char, N>(platform, n, repeat, report_timing);
    run_test<short, N>(platform, n, repeat, report_timing);
    run_test<int, N>(platform, n, repeat, report_timing);
    run_test<long, N>(platform, n, repeat, report_timing);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  const int64_t n = std::atol(argv[1]);
  const int repeat = std::atoi(argv[2]);
  run<128>(platform, n, repeat);
  run<256>(platform, n, repeat);
  run<512>(platform, n, repeat);
  run<1024>(platform, n, repeat);
  run<2048>(platform, n, repeat);
  platform.finalize();
  return 0;
}
