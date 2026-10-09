#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 256;

bool parse_positive_int(const char* text, int* value)
{
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

uint32_t wrap_sum_partials(const std::vector<int>& partials)
{
  uint32_t total = 0;
  for (int value : partials) {
    total += static_cast<uint32_t>(value);
  }
  return total;
}

} // namespace

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  int n = 0;
  int repeat = 0;
  if (!parse_positive_int(argv[1], &n) || !parse_positive_int(argv[2], &repeat)) {
    std::fprintf(stderr, "arguments must be positive 32-bit integers\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int blocks = (n + kThreads - 1) / kThreads;
  std::vector<int> data(static_cast<size_t>(n), 0);
  std::vector<int> reduce_partials(static_cast<size_t>(blocks), 0);
  std::vector<int> scan_partials(static_cast<size_t>(blocks), 0);

  iris::DMem mem_data(data.data(), data.size() * sizeof(int));
  iris::DMem mem_reduce(reduce_partials.data(), reduce_partials.size() * sizeof(int));
  iris::DMem mem_scan(scan_partials.data(), scan_partials.size() * sizeof(int));

  uint32_t reduce_total = UINT32_MAX;
  uint32_t scan_last = 0;

  const size_t lws[1] = { kThreads };
  const size_t gws_data[1] = { static_cast<size_t>(blocks) * kThreads };
  const size_t gws_partials[1] = { static_cast<size_t>(blocks) * kThreads };

  auto start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < repeat; ++iter) {
    {
      iris::Task task;
      void* params[2] = { &mem_data, &n };
      int pinfo[2] = { iris_w, sizeof(n) };
      task.kernel("sequence_kernel", 1, nullptr, const_cast<size_t*>(gws_data),
                  const_cast<size_t*>(lws), 2, params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
    }

    {
      iris::Task task;
      void* params[4] = { &mem_data, &mem_reduce, &mem_scan, &n };
      int pinfo[4] = { iris_r, iris_w, iris_w, sizeof(n) };
      task.kernel("partial_sum_kernel", 1, nullptr, const_cast<size_t*>(gws_partials),
                  const_cast<size_t*>(lws), 4, params, pinfo);
      task.flush_out(mem_reduce);
      task.flush_out(mem_scan);
      task.submit(iris_gpu, nullptr, 1);
    }

    mem_reduce.fetch(reduce_partials.data());
    mem_scan.fetch(scan_partials.data());
    reduce_total = wrap_sum_partials(reduce_partials);
    scan_last = wrap_sum_partials(scan_partials);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time: " << (time * 1e-3f) / repeat << " (us)\n";

  const uint32_t diff = reduce_total - scan_last;
  std::cout << ((diff == 0) ? "PASS" : "FAIL") << "\n";

  platform.finalize();
  return diff == 0 ? 0 : 1;
}
