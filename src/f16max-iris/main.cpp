#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define NUM_OF_BLOCKS 1048576
#define NUM_OF_THREADS 256

struct Half2 {
  _Float16 x;
  _Float16 y;
};

static_assert(sizeof(Half2) == 4, "Half2 layout must match HIP half2");

static void generateInput(Half2* a, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    a[i].x = static_cast<_Float16>(static_cast<float>(std::rand() % 922021));
    a[i].y = static_cast<_Float16>(static_cast<float>(std::rand() % 922021));
  }
}

static int run_kernel(
    const char* kernel_name,
    iris::DMem& mem_a,
    iris::DMem& mem_b,
    iris::DMem& mem_r,
    size_t work_items,
    int repeat,
    bool flush_result,
    bool print_time)
{
  const size_t gws[1] = { static_cast<size_t>(NUM_OF_BLOCKS) * NUM_OF_THREADS };
  const size_t lws[1] = { NUM_OF_THREADS };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = { &mem_a, &mem_b, &mem_r, &work_items };
    int pinfo[4] = { iris_r, iris_r, iris_w, static_cast<int>(sizeof(work_items)) };
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                4, params, pinfo);
    if (flush_result && i == repeat - 1) task.flush_out(mem_r);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (print_time) {
    std::printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);
  }
  return 0;
}

static bool verify_half2(const Half2* a, const Half2* b, const Half2* r, size_t size, float tol)
{
  for (size_t i = 0; i < size; ++i) {
    const float ax = static_cast<float>(a[i].x);
    const float ay = static_cast<float>(a[i].y);
    const float bx = static_cast<float>(b[i].x);
    const float by = static_cast<float>(b[i].y);
    const float rx = static_cast<float>(r[i].x);
    const float ry = static_cast<float>(r[i].y);
    if (std::fabs(rx - std::fmax(ax, bx)) > tol || std::fabs(ry - std::fmax(ay, by)) > tol) {
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const size_t size = static_cast<size_t>(NUM_OF_BLOCKS) * NUM_OF_THREADS;
  const size_t size_bytes = size * sizeof(Half2);

  Half2* a = static_cast<Half2*>(std::malloc(size_bytes));
  Half2* b = static_cast<Half2*>(std::malloc(size_bytes));
  Half2* r = static_cast<Half2*>(std::malloc(size_bytes));
  if (!a || !b || !r) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(a);
    std::free(b);
    std::free(r);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  generateInput(a, size);
  generateInput(b, size);

  iris::DMem mem_a(a, size_bytes);
  iris::DMem mem_b(b, size_bytes);
  iris::DMem mem_r(r, size_bytes);

  run_kernel("hmax_half2", mem_a, mem_b, mem_r, size, repeat, false, false);
  run_kernel("hmax_half2", mem_a, mem_b, mem_r, size, repeat, true, true);
  bool ok = verify_half2(a, b, r, size, 1e-2f);
  std::printf("fp16_hmax2 %s\n", ok ? "PASS" : "FAIL");

  const size_t scalar_size = size * 2;
  run_kernel("hmax_half", mem_a, mem_b, mem_r, scalar_size, repeat, false, false);
  run_kernel("hmax_half", mem_a, mem_b, mem_r, scalar_size, repeat, true, true);
  const bool ok_scalar = verify_half2(a, b, r, size, 1e-3f);
  std::printf("fp16_hmax %s\n", ok_scalar ? "PASS" : "FAIL");

  std::free(a);
  std::free(b);
  std::free(r);
  platform.finalize();
  return (ok && ok_scalar) ? 0 : 1;
}
