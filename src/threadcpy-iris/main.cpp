#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

static constexpr int kBlockSize = 256;

template <typename scalar_t>
static int test_threads_copy(const char* kernel_name, size_t n, int vec_size, int repeat) {
  auto* in_h = new scalar_t[n];
  auto* out_h = new scalar_t[n];
  for (size_t i = 0; i < n; i++) {
    in_h[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    out_h[i] = 0;
  }

  const size_t bytes = n * sizeof(scalar_t);
  iris::DMem mem_in(in_h, bytes);
  iris::DMem mem_out(out_h, bytes);

  const size_t block_work_size = static_cast<size_t>(kBlockSize) * static_cast<size_t>(vec_size);
  const size_t blocks = (n + block_work_size - 1) / block_work_size;
  size_t gws[1] = { blocks * kBlockSize };
  size_t lws[1] = { kBlockSize };

  for (int i = 0; i < 100; i++) {
    iris::Task task;
    void* params[3] = { &mem_in, &mem_out, &n };
    int pinfo[3] = { iris_r, iris_w, sizeof(n) };
    task.kernel(kernel_name, 1, nullptr, gws, lws, 3, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }

  mem_out.fetch(out_h);
  const int cmp = std::memcmp(out_h, in_h, bytes);
  std::cout << (cmp ? "FAIL" : "PASS") << std::endl;

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = { &mem_in, &mem_out, &n };
    int pinfo[3] = { iris_r, iris_w, sizeof(n) };
    task.kernel(kernel_name, 1, nullptr, gws, lws, 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  mem_out.fetch(out_h);
  auto end = std::chrono::steady_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  float avg_time_ms = time * 1e-6f / repeat;
  std::cout << "Average kernel execution time (ms):" << avg_time_ms << " Throughput:";

  float total_GBytes = 2 * n * sizeof(scalar_t) / 1000.0f / 1000.0f;
  std::cout << total_GBytes / avg_time_ms << " GB/s" << std::endl;

  delete[] in_h;
  delete[] out_h;
  return cmp ? 1 : 0;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <repeat>" << std::endl;
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);

  std::cout << "1GB threads copy test ..." << std::endl;
  const size_t numel = 1024 * 1024 * 256 + 2;
  std::srand(19937);

  int errors = 0;
  std::cout << "int1: ";
  errors += test_threads_copy<int>("threads_copy_int1", numel, 1, repeat);
  std::cout << "int2: ";
  errors += test_threads_copy<int>("threads_copy_int2", numel, 2, repeat);
  std::cout << "int4: ";
  errors += test_threads_copy<int>("threads_copy_int4", numel, 4, repeat);
  std::cout << "int8: ";
  errors += test_threads_copy<int>("threads_copy_int8", numel, 8, repeat);
  std::cout << "int16: ";
  errors += test_threads_copy<int>("threads_copy_int16", numel, 16, repeat);

  std::cout << "short1: ";
  errors += test_threads_copy<short>("threads_copy_short1", numel * 2, 1, repeat);
  std::cout << "short2: ";
  errors += test_threads_copy<short>("threads_copy_short2", numel * 2, 2, repeat);
  std::cout << "short4: ";
  errors += test_threads_copy<short>("threads_copy_short4", numel * 2, 4, repeat);
  std::cout << "short8: ";
  errors += test_threads_copy<short>("threads_copy_short8", numel * 2, 8, repeat);
  std::cout << "short16: ";
  errors += test_threads_copy<short>("threads_copy_short16", numel * 2, 16, repeat);

  std::cout << "char1: ";
  errors += test_threads_copy<char>("threads_copy_char1", numel * 4, 1, repeat);
  std::cout << "char2: ";
  errors += test_threads_copy<char>("threads_copy_char2", numel * 4, 2, repeat);
  std::cout << "char4: ";
  errors += test_threads_copy<char>("threads_copy_char4", numel * 4, 4, repeat);
  std::cout << "char8: ";
  errors += test_threads_copy<char>("threads_copy_char8", numel * 4, 8, repeat);
  std::cout << "char16: ";
  errors += test_threads_copy<char>("threads_copy_char16", numel * 4, 16, repeat);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}
