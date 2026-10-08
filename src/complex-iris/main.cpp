#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static bool check(const char* cs, int n)
{
  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (cs[i] != 5) {
      ok = false;
      break;
    }
  }
  return ok;
}

static double run_kernel(const char* name, iris::DMem& mem_cs, int n, int repeat)
{
  const size_t gws[1] = { static_cast<size_t>((n + 255) / 256) * 256 };
  const size_t lws[1] = { 256 };
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = { &mem_cs, &n };
    int pinfo[2] = { iris_w, sizeof(n) };
    task.kernel(name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 2, params, pinfo);
    task.flush_out(mem_cs);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return time * 1e-9 / repeat;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <problem size> <repeat>\n", argv[0]);
    return 1;
  }
  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  char* cs = static_cast<char*>(std::malloc(n));
  if (!cs) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }
  iris::DMem mem_cs(cs, n);

  run_kernel("complex_float", mem_cs, n, 1);
  run_kernel("complex_double", mem_cs, n, 1);

  std::printf("\nSingle-precision complex data type\n");
  double time = run_kernel("complex_float", mem_cs, n, repeat);
  std::printf("Average kernel execution time %f (s)\n", time);
  mem_cs.fetch(cs);
  bool complex_float_check = check(cs, n);

  std::printf("\nDouble-precision complex data type\n");
  time = run_kernel("complex_double", mem_cs, n, repeat);
  std::printf("Average kernel execution time %f (s)\n", time);
  mem_cs.fetch(cs);
  bool complex_double_check = check(cs, n);

  std::printf("%s\n", (complex_float_check && complex_double_check) ? "PASS" : "FAIL");
  std::free(cs);
  platform.finalize();
  return (complex_float_check && complex_double_check) ? 0 : 1;
}
