#include <iris/iris.hpp>

#include "common.h"
#include "reference.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

static void validate_result(
    const float* gpu_result,
    const float* cpu_reference,
    const char* name,
    std::uint64_t num_elements,
    float tolerance=1e-4,
    int n_print=5)
{
  int nfaults = 0;
  for (uint64_t i = 0; i < num_elements; i++) {
    if (std::fabs(cpu_reference[i] - gpu_result[i]) > tolerance &&
        std::isfinite(cpu_reference[i])) {
      std::printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n",
                  name, static_cast<size_t>(i), cpu_reference[i], gpu_result[i]);
      nfaults++;
      if (nfaults >= max_int(10, n_print)) break;
    }
  }

  std::printf("%s\n", (nfaults == 0) ? "PASS" : "FAIL");
}

static float benchmark_forward(
    int repeat,
    const char* kernel_name,
    iris::DMem& mem_x,
    iris::DMem& mem_out,
    std::uint64_t N,
    int block_size)
{
  const bool vec_kernel = kernel_name[19] == '2';
  std::uint64_t work_items = vec_kernel ? N / 4 : N;
  size_t gws[1] = {static_cast<size_t>(ceil_div(work_items, block_size) * block_size)};
  size_t lws[1] = {static_cast<size_t>(block_size)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = {&mem_x, &mem_out, &N};
    int pinfo[3] = {iris_r, iris_w, sizeof(N)};
    task.kernel(kernel_name, 1, nullptr, gws, lws, 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-6f) / repeat;
}

static float benchmark_backward(
    int repeat,
    const char* kernel_name,
    iris::DMem& mem_dout,
    iris::DMem& mem_x,
    iris::DMem& mem_dx,
    std::uint64_t N,
    int block_size)
{
  const bool vec_kernel = kernel_name[20] == '3';
  std::uint64_t work_items = vec_kernel ? N / 4 : N;
  size_t gws[1] = {static_cast<size_t>(ceil_div(work_items, block_size) * block_size)};
  size_t lws[1] = {static_cast<size_t>(block_size)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {&mem_dout, &mem_x, &mem_dx, &N};
    int pinfo[4] = {iris_r, iris_r, iris_w, sizeof(N)};
    task.kernel(kernel_name, 1, nullptr, gws, lws, 4, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-6f) / repeat;
}

static void run_forward(
    const char* kernel_name,
    iris::DMem& mem_x,
    iris::DMem& mem_out,
    std::uint64_t N,
    int block_size,
    bool flush)
{
  const bool vec_kernel = kernel_name[19] == '2';
  std::uint64_t work_items = vec_kernel ? N / 4 : N;
  size_t gws[1] = {static_cast<size_t>(ceil_div(work_items, block_size) * block_size)};
  size_t lws[1] = {static_cast<size_t>(block_size)};
  iris::Task task;
  void* params[3] = {&mem_x, &mem_out, &N};
  int pinfo[3] = {iris_r, iris_w, sizeof(N)};
  task.kernel(kernel_name, 1, nullptr, gws, lws, 3, params, pinfo);
  if (flush) task.flush_out(mem_out);
  task.submit(iris_gpu, nullptr, 1);
}

static void run_backward(
    const char* kernel_name,
    iris::DMem& mem_dout,
    iris::DMem& mem_x,
    iris::DMem& mem_dx,
    std::uint64_t N,
    int block_size,
    bool flush)
{
  const bool vec_kernel = kernel_name[20] == '3';
  std::uint64_t work_items = vec_kernel ? N / 4 : N;
  size_t gws[1] = {static_cast<size_t>(ceil_div(work_items, block_size) * block_size)};
  size_t lws[1] = {static_cast<size_t>(block_size)};
  iris::Task task;
  void* params[4] = {&mem_dout, &mem_x, &mem_dx, &N};
  int pinfo[4] = {iris_r, iris_r, iris_w, sizeof(N)};
  task.kernel(kernel_name, 1, nullptr, gws, lws, 4, params, pinfo);
  if (flush) task.flush_out(mem_dx);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::printf("Usage: %s <batch size> <number of channels> <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  uint64_t B = std::atoi(argv[1]);
  uint64_t C = std::atoi(argv[2]);
  uint64_t H = std::atoi(argv[3]);
  uint64_t W = std::atoi(argv[4]);
  int repeat = std::atoi(argv[5]);
  uint64_t N = B * C * H * W;

  srand(0);
  float* x = make_random_float(N);
  float* dout = make_random_float(N);
  float* out = static_cast<float*>(std::malloc(N * sizeof(float)));
  float* dx = static_cast<float*>(std::malloc(N * sizeof(float)));
  float* gpu = static_cast<float*>(std::malloc(N * sizeof(float)));

  silu_forward_reference(x, out, N);
  silu_backward_reference(dout, x, dx, N);

  iris::DMem mem_x(x, N * sizeof(float));
  iris::DMem mem_out(gpu, N * sizeof(float));
  iris::DMem mem_dout(dout, N * sizeof(float));
  iris::DMem mem_dx(gpu, N * sizeof(float));

  int block_sizes[] = {64, 128, 256, 512, 1024};
  std::printf("Checking forward pass\n");
  for (int block_size: block_sizes) {
    std::printf("Checking block size %d\n", block_size);
    run_forward("silu_forward_kernel", mem_x, mem_out, N, block_size, true);
    mem_out.fetch(gpu);
    validate_result(gpu, out, "out", N);
  }

  std::printf("Checking forward2 pass\n");
  for (int block_size: block_sizes) {
    std::printf("Checking block size %d\n", block_size);
    run_forward("silu_forward_kernel2", mem_x, mem_out, N, block_size, true);
    mem_out.fetch(gpu);
    validate_result(gpu, out, "out", N);
  }

  std::printf("Checking backward pass\n");
  for (int block_size: block_sizes) {
    std::printf("Checking block size %d\n", block_size);
    run_backward("silu_backward_kernel", mem_dout, mem_x, mem_dx, N, block_size, true);
    mem_dx.fetch(gpu);
    validate_result(gpu, dx, "dx", N);
  }
  std::printf("Checking backward2 pass\n");
  for (int block_size: block_sizes) {
    std::printf("Checking block size %d\n", block_size);
    run_backward("silu_backward_kernel2", mem_dout, mem_x, mem_dx, N, block_size, true);
    mem_dx.fetch(gpu);
    validate_result(gpu, dx, "dx", N);
  }
  std::printf("Checking backward3 pass\n");
  for (int block_size: block_sizes) {
    std::printf("Checking block size %d\n", block_size);
    run_backward("silu_backward_kernel3", mem_dout, mem_x, mem_dx, N, block_size, true);
    mem_dx.fetch(gpu);
    validate_result(gpu, dx, "dx", N);
  }

  std::printf("\nForward pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_forward(repeat, "silu_forward_kernel", mem_x, mem_out, N, block_size);
    std::printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  std::printf("\nForward2 pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_forward(repeat, "silu_forward_kernel2", mem_x, mem_out, N, block_size);
    std::printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  std::printf("\nBackward pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_backward(repeat, "silu_backward_kernel", mem_dout, mem_x, mem_dx, N, block_size);
    std::printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  std::printf("\nBackward2 pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_backward(repeat, "silu_backward_kernel2", mem_dout, mem_x, mem_dx, N, block_size);
    std::printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  std::printf("\nBackward3 pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_backward(repeat, "silu_backward_kernel3", mem_dout, mem_x, mem_dx, N, block_size);
    std::printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  std::free(x);
  std::free(out);
  std::free(dout);
  std::free(dx);
  std::free(gpu);
  platform.finalize();
  return 0;
}
