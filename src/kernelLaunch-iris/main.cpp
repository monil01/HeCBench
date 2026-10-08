#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static void initArgs(char* args, size_t n, char value)
{
  for (size_t i = 0; i < n; i++) {
    args[i] = static_cast<char>(value + (i & 7));
  }
}

static double runKernel(const char* kernel_name, iris::DMem& mem_args, iris::DMem& mem_out, int repeat)
{
  constexpr size_t one = 1;
  for (int i = 0; i < repeat; i++) {
    iris::Task warmup;
    void* params[2] = {&mem_args, &mem_out};
    int pinfo[2] = {iris_r, iris_w};
    warmup.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(&one),
                  const_cast<size_t*>(&one), 2, params, pinfo);
    warmup.submit(iris_gpu, nullptr, 1);
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = {&mem_args, &mem_out};
    int pinfo[2] = {iris_r, iris_w};
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(&one),
                const_cast<size_t*>(&one), 2, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-3) / repeat;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  if (repeat <= 0) {
    std::fprintf(stderr, "repeat must be positive\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  char small_kernel_args[16]{};
  char medium_kernel_args[256]{};
  char large_kernel_args[4096]{};
  initArgs(small_kernel_args, sizeof(small_kernel_args), 1);
  initArgs(medium_kernel_args, sizeof(medium_kernel_args), 2);
  initArgs(large_kernel_args, sizeof(large_kernel_args), 3);

  char out = 0;
  iris::DMem mem_small(small_kernel_args, sizeof(small_kernel_args));
  iris::DMem mem_medium(medium_kernel_args, sizeof(medium_kernel_args));
  iris::DMem mem_large(large_kernel_args, sizeof(large_kernel_args));
  iris::DMem mem_out(&out, sizeof(out));

  const double small_us = runKernel("KernelWithSmallArgs", mem_small, mem_out, repeat);
  mem_out.fetch(&out);
  const bool small_ok = out == small_kernel_args[0];
  std::printf("Average execution time of kernelWithSmallArgs: %f (us)\n", small_us);

  out = 0;
  mem_out.update(&out);
  const double medium_us = runKernel("KernelWithMediumArgs", mem_medium, mem_out, repeat);
  mem_out.fetch(&out);
  const bool medium_ok = out == medium_kernel_args[0];
  std::printf("Average execution time of kernelWithMediumArgs: %f (us)\n", medium_us);

  out = 0;
  mem_out.update(&out);
  const double large_us = runKernel("KernelWithLargeArgs", mem_large, mem_out, repeat);
  mem_out.fetch(&out);
  const bool large_ok = out == large_kernel_args[0];
  std::printf("Average execution time of kernelWithLargeArgs: %f (us)\n", large_us);

  std::printf("%s\n", (small_ok && medium_ok && large_ok) ? "PASS" : "FAIL");

  platform.finalize();
  return (small_ok && medium_ok && large_ok) ? 0 : 1;
}
