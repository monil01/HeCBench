#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using host_half = _Float16;

static void fill_values(float* r, host_half* a, host_half* b, size_t n)
{
  for (size_t i = 0; i < n; ++i) {
    float ta = rand() % 2 ? -1.0f : 1.0f;
    float tb = rand() % 2 ? -1.0f : 1.0f;
    r[i] = ta + tb;
    a[i] = ta;
    b[i] = tb;
  }
}

static bool allclose(const float* ref, const host_half* got, size_t n, float atol = 1e-2f)
{
  for (size_t i = 0; i < n; ++i) {
    float r = ref[i];
    float g = static_cast<float>(got[i]);
    if (std::fabs(r - g) > atol + 1e-5f * std::fabs(r)) {
      std::fprintf(stderr, "  mismatch at [%zu]: ref=%.4f  got=%.4f\n", i, r, g);
      return false;
    }
  }
  return true;
}

struct BenchResult {
  float avg_ms;
  float gbps;
};

static void print_result(const char* label, bool pass, BenchResult r, int repeats)
{
  (void)label;
  std::printf("  correctness : %s\n", pass ? "PASS" : "FAIL");
  std::printf("  avg  time   : %.3f ms  (%d iters)\n", r.avg_ms, repeats);
  std::printf("  bandwidth   : %.1f GB/s\n\n", r.gbps);
}

static BenchResult run_kernel(const char* name, iris::DMem& mem_a, iris::DMem& mem_b,
                              iris::DMem& mem_c, host_half* h_out, size_t total,
                              int m, int n, int repeat)
{
  const size_t total_bytes = total * sizeof(host_half);
  const double bytes_moved = 3.0 * total_bytes;
  std::memset(h_out, 0, total_bytes);

  auto submit_once = [&](bool flush) {
    iris::Task task;
    if (std::strcmp(name, "tv_elementwise_add_kernel") == 0) {
      const size_t gws[2] = {static_cast<size_t>(n), static_cast<size_t>(m)};
      const size_t lws[2] = {16, 16};
      void* params[5] = {&mem_a, &mem_b, &mem_c, &m, &n};
      int pinfo[5] = {iris_r, iris_r, iris_w, sizeof(m), sizeof(n)};
      task.kernel(name, 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                  5, params, pinfo);
    } else {
      const size_t lws[1] = {256};
      const size_t gws[1] = {static_cast<size_t>(((total + 255) / 256) * 256)};
      void* params[4] = {&mem_a, &mem_b, &mem_c, &total};
      int pinfo[4] = {iris_r, iris_r, iris_w, sizeof(total)};
      task.kernel(name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                  4, params, pinfo);
    }
    if (flush) task.flush_out(mem_c);
    task.submit(iris_gpu, nullptr, 1);
  };

  submit_once(true);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) submit_once(false);
  iris::Task flush;
  flush.flush_out(mem_c);
  flush.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  float ms = elapsed * 1e-6f / repeat;
  float gbps = static_cast<float>(bytes_moved / (ms * 1e-3) / 1e9);
  return {ms, gbps};
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int m = std::atoi(argv[1]);
  const int n = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  const size_t total = static_cast<size_t>(m) * n;
  const size_t bytes = total * sizeof(host_half);

  host_half* h_a = static_cast<host_half*>(std::malloc(bytes));
  host_half* h_b = static_cast<host_half*>(std::malloc(bytes));
  host_half* h_out = static_cast<host_half*>(std::malloc(bytes));
  float* h_ref = static_cast<float*>(std::malloc(total * sizeof(float)));
  if (!h_a || !h_b || !h_out || !h_ref) {
    std::fprintf(stderr, "allocation failure\n");
    platform.finalize();
    return 1;
  }

  std::srand(42);
  fill_values(h_ref, h_a, h_b, total);

  iris::DMem mem_a(h_a, bytes);
  iris::DMem mem_b(h_b, bytes);
  iris::DMem mem_c(h_out, bytes);

  std::printf("=== Base (one thread per element) ===\n");
  BenchResult r1 = run_kernel("base_elementwise_add_kernel", mem_a, mem_b, mem_c,
                              h_out, total, m, n, repeat);
  bool pass1 = allclose(h_ref, h_out, total);
  print_result("base", pass1, r1, repeat);

  std::printf("=== Vectorised (8x fp16 = 128-bit load per thread) ===\n");
  BenchResult r2 = run_kernel("vectorized_elementwise_add_kernel", mem_a, mem_b, mem_c,
                              h_out, total, m, n, repeat);
  bool pass2 = allclose(h_ref, h_out, total);
  print_result("vectorized", pass2, r2, repeat);

  std::printf("=== TV-layout (block tile 16x256, 8x fp16 per thread per row) ===\n");
  assert(m % 16 == 0);
  assert(n % 256 == 0);
  BenchResult r3 = run_kernel("tv_elementwise_add_kernel", mem_a, mem_b, mem_c,
                              h_out, total, m, n, repeat);
  bool pass3 = allclose(h_ref, h_out, total);
  print_result("tv_layout", pass3, r3, repeat);

  std::printf("-----------------------------------------------------\n");
  std::printf("%-12s  %9s  %10s\n", "Kernel", "Avg(ms)", "BW(GB/s)");
  std::printf("-----------------------------------------------------\n");
  std::printf("%-12s  %9.3f  %10.1f\n", "base", r1.avg_ms, r1.gbps);
  std::printf("%-12s  %9.3f  %10.1f\n", "vectorized", r2.avg_ms, r2.gbps);
  std::printf("%-12s  %9.3f  %10.1f\n", "tv_layout", r3.avg_ms, r3.gbps);
  std::printf("-----------------------------------------------------\n");

  std::free(h_a);
  std::free(h_b);
  std::free(h_ref);
  std::free(h_out);
  platform.finalize();
  return (pass1 && pass2 && pass3) ? 0 : 1;
}
