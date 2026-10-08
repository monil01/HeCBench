#include <iris/iris.hpp>

#include <chrono>
#include <climits>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#define NUM_BLOCKS 1024
#define BLOCK_SIZE 256

template <typename T>
static void run_kernel(const char* kernel_name, T* h_ptr, iris::DMem& mem, int repeat, const char* label)
{
  mem.update(h_ptr);

  const size_t gws[1] = {NUM_BLOCKS * BLOCK_SIZE};
  const size_t lws[1] = {BLOCK_SIZE};

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[1] = {&mem};
    int pinfo[1] = {iris_rw};
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 1, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  iris::Task flush;
  flush.flush_out(mem);
  flush.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("%s | Average execution time: %f (s)\n", label, (time * 1e-9f) / repeat);

  mem.fetch(h_ptr);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);

  unsigned long long res_u64[3] = {ULONG_MAX, 0, 0};
  long long res_s64[3] = {LONG_MAX, LONG_MIN, 0};
  double res_f64[3] = {DBL_MAX, DBL_MIN, 0};

  iris::DMem mem_u64_min(res_u64, sizeof(unsigned long long));
  iris::DMem mem_s64_min(res_s64, sizeof(long long));
  iris::DMem mem_f64_min(res_f64, sizeof(double));
  iris::DMem mem_u64_max(res_u64 + 1, sizeof(unsigned long long));
  iris::DMem mem_s64_max(res_s64 + 1, sizeof(long long));
  iris::DMem mem_f64_max(res_f64 + 1, sizeof(double));
  iris::DMem mem_u64_add(res_u64 + 2, sizeof(unsigned long long));
  iris::DMem mem_s64_add(res_s64 + 2, sizeof(long long));
  iris::DMem mem_f64_add(res_f64 + 2, sizeof(double));

  run_kernel("atomic_min_u64", res_u64, mem_u64_min, repeat, "Atomic min for data type U64");
  run_kernel("atomic_min_s64", res_s64, mem_s64_min, repeat, "Atomic min for data type S64");
  run_kernel("atomic_min_f64", res_f64, mem_f64_min, repeat, "Atomic min for data type F64");

  run_kernel("atomic_max_u64", res_u64 + 1, mem_u64_max, repeat, "Atomic max for data type U64");
  run_kernel("atomic_max_s64", res_s64 + 1, mem_s64_max, repeat, "Atomic max for data type S64");
  run_kernel("atomic_max_f64", res_f64 + 1, mem_f64_max, repeat, "Atomic max for data type F64");

  run_kernel("atomic_add_u64", res_u64 + 2, mem_u64_add, 1, "Atomic add for data type U64");
  run_kernel("atomic_add_s64", res_s64 + 2, mem_s64_add, 1, "Atomic add for data type S64");
  run_kernel("atomic_add_f64", res_f64 + 2, mem_f64_add, 1, "Atomic add for data type F64");

  const unsigned long long bound = NUM_BLOCKS * BLOCK_SIZE;
  unsigned long long sum = 0;
  for (unsigned int i = 1; i <= bound; i++) sum += i;

  bool error = false;
  if (res_u64[0] != 1ULL || res_s64[0] != 1LL || res_f64[0] != 1.0) {
    error = true;
    std::printf("atomic min results: %llu %lld %lf\n", res_u64[0], res_s64[0], res_f64[0]);
  }
  if (res_u64[1] != bound || res_s64[1] != static_cast<long long>(bound) ||
      res_f64[1] != static_cast<double>(bound)) {
    error = true;
    std::printf("atomic max results: %llu %lld %lf\n", res_u64[1], res_s64[1], res_f64[1]);
  }
  if (res_u64[2] != sum || res_s64[2] != static_cast<long long>(sum) ||
      res_f64[2] != static_cast<double>(sum)) {
    error = true;
    std::printf("atomic add results: %llu %lld %lf\n", res_u64[2], res_s64[2], res_f64[2]);
  }
  std::printf("%s\n", error ? "FAIL" : "PASS");

  platform.finalize();
  return error ? 1 : 0;
}
