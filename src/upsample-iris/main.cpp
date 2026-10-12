#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "reference.h"

static int ceil_div_int(int dividend, int divisor) {
  return (dividend + divisor - 1) / divisor;
}

static float* make_random_float(size_t n) {
  float* arr = static_cast<float*>(std::malloc(n * sizeof(float)));
  if (!arr) {
    std::fprintf(stderr, "malloc failed\n");
    std::exit(1);
  }
  for (size_t i = 0; i < n; i++) {
    arr[i] = std::rand() / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
  }
  return arr;
}

static bool validate_result(const float* gpu, const float* cpu_reference, const char* name,
                            uint64_t num_elements, float tolerance = 1e-4f) {
  int nfaults = 0;
  for (uint64_t i = 0; i < num_elements; i++) {
    if (std::fabs(cpu_reference[i] - gpu[i]) > tolerance && std::isfinite(cpu_reference[i])) {
      std::printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name,
                  static_cast<size_t>(i), cpu_reference[i], gpu[i]);
      nfaults++;
      if (nfaults >= 10) break;
    }
  }
  std::printf("%s\n", (nfaults == 0) ? "PASS" : "FAIL");
  return nfaults == 0;
}

static double submit_forward1(iris::DMem& mem_x, iris::DMem& mem_out, int B, int C, int H,
                              int W, int block_size, bool flush) {
  size_t total = static_cast<size_t>(B) * C * H * W;
  size_t gws[1] = {static_cast<size_t>(ceil_div_int(static_cast<int>(total), block_size) *
                                       block_size)};
  size_t lws[1] = {static_cast<size_t>(block_size)};
  void* params[] = {&mem_x, &mem_out, &B, &C, &H, &W};
  int pinfo[] = {iris_r, iris_w, sizeof(B), sizeof(C), sizeof(H), sizeof(W)};
  iris::Task task;
  task.kernel("upsample_forward_kernel", 1, nullptr, gws, lws, 6, params, pinfo);
  if (flush) task.flush_out(mem_out);
  auto start = std::chrono::steady_clock::now();
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
}

static double submit_backward1(iris::DMem& mem_dout, iris::DMem& mem_dx, int B, int C, int H,
                               int W, int block_size, bool flush) {
  size_t total = static_cast<size_t>(B) * C * H * W;
  size_t gws[1] = {static_cast<size_t>(ceil_div_int(static_cast<int>(total), block_size) *
                                       block_size)};
  size_t lws[1] = {static_cast<size_t>(block_size)};
  void* params[] = {&mem_dout, &mem_dx, &B, &C, &H, &W};
  int pinfo[] = {iris_r, iris_w, sizeof(B), sizeof(C), sizeof(H), sizeof(W)};
  iris::Task task;
  task.kernel("upsample_backward_kernel", 1, nullptr, gws, lws, 6, params, pinfo);
  if (flush) task.flush_out(mem_dx);
  auto start = std::chrono::steady_clock::now();
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
}

static double submit_forward2(iris::DMem& mem_x, iris::DMem& mem_out, int B, int C, int H,
                              int W, int block_size, bool flush) {
  size_t gws[3] = {static_cast<size_t>(ceil_div_int(W, block_size) * block_size),
                   static_cast<size_t>(ceil_div_int(H, block_size) * block_size),
                   static_cast<size_t>(B * C)};
  size_t lws[3] = {static_cast<size_t>(block_size), static_cast<size_t>(block_size), 1};
  void* params[] = {&mem_x, &mem_out, &B, &C, &H, &W};
  int pinfo[] = {iris_r, iris_w, sizeof(B), sizeof(C), sizeof(H), sizeof(W)};
  iris::Task task;
  task.kernel("upsample_forward_kernel2", 3, nullptr, gws, lws, 6, params, pinfo);
  if (flush) task.flush_out(mem_out);
  auto start = std::chrono::steady_clock::now();
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
}

static double submit_backward2(iris::DMem& mem_dout, iris::DMem& mem_dx, int B, int C, int H,
                               int W, int block_size, bool flush) {
  size_t gws[3] = {static_cast<size_t>(ceil_div_int(W, block_size) * block_size),
                   static_cast<size_t>(ceil_div_int(H, block_size) * block_size),
                   static_cast<size_t>(B * C)};
  size_t lws[3] = {static_cast<size_t>(block_size), static_cast<size_t>(block_size), 1};
  void* params[] = {&mem_dout, &mem_dx, &B, &C, &H, &W};
  int pinfo[] = {iris_r, iris_w, sizeof(B), sizeof(C), sizeof(H), sizeof(W)};
  iris::Task task;
  task.kernel("upsample_backward_kernel2", 3, nullptr, gws, lws, 6, params, pinfo);
  if (flush) task.flush_out(mem_dx);
  auto start = std::chrono::steady_clock::now();
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::printf("Usage: %s <batch size> <number of channels> <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int B = std::atoi(argv[1]);
  int C = std::atoi(argv[2]);
  int H = std::atoi(argv[3]);
  int W = std::atoi(argv[4]);
  int repeat = std::atoi(argv[5]);

  std::srand(0);
  uint64_t S = static_cast<uint64_t>(B) * C * H * W;
  float* x = make_random_float(S);
  float* dout = make_random_float(S * 4);
  float* out = static_cast<float*>(std::malloc(S * 4 * sizeof(float)));
  float* dx = static_cast<float*>(std::malloc(S * sizeof(float)));
  float* gpu_out = static_cast<float*>(std::malloc(S * 4 * sizeof(float)));
  float* gpu_dx = static_cast<float*>(std::malloc(S * sizeof(float)));
  if (!out || !dx || !gpu_out || !gpu_dx) return 1;

  upsample_forward_reference(x, out, B, C, H, W);
  upsample_backward_reference(dout, dx, B, C, H, W);

  iris::DMem mem_x(x, S * sizeof(float));
  iris::DMem mem_dout(dout, S * 4 * sizeof(float));
  iris::DMem mem_out(gpu_out, S * 4 * sizeof(float));
  iris::DMem mem_dx(gpu_dx, S * sizeof(float));

  bool ok = true;
  int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  std::printf("Checking forward pass\n");
  for (int block_size : block_sizes) {
    std::printf("Checking block size %d\n", block_size);
    submit_forward1(mem_x, mem_out, B, C, H, W, block_size, true);
    ok = validate_result(gpu_out, out, "out", S * 4) && ok;
  }

  std::printf("Forward1 pass benchmarks:\n");
  for (int block_size : block_sizes) {
    double elapsed = 0.0;
    for (int i = 0; i < repeat; i++) elapsed += submit_forward1(mem_x, mem_out, B, C, H, W, block_size, false);
    elapsed /= repeat;
    float gflops = static_cast<float>(S) / elapsed * 1e3f / 1e9f;
    std::printf("block_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed, gflops);
  }

  int block2D_sizes[] = {8, 16, 32};
  for (int block_size : block2D_sizes) {
    std::printf("Checking block size %d\n", block_size);
    submit_forward2(mem_x, mem_out, B, C, H, W, block_size, true);
    ok = validate_result(gpu_out, out, "out", S * 4) && ok;
  }

  std::printf("Forward2 pass benchmarks:\n");
  for (int block_size : block2D_sizes) {
    double elapsed = 0.0;
    for (int i = 0; i < repeat; i++) elapsed += submit_forward2(mem_x, mem_out, B, C, H, W, block_size, false);
    elapsed /= repeat;
    float gflops = static_cast<float>(S) / elapsed * 1e3f / 1e9f;
    std::printf("block2D_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed, gflops);
  }

  std::printf("Checking backward pass\n");
  for (int block_size : block_sizes) {
    std::printf("Checking block size %d\n", block_size);
    submit_backward1(mem_dout, mem_dx, B, C, H, W, block_size, true);
    ok = validate_result(gpu_dx, dx, "dx", S) && ok;
  }

  std::printf("Backward pass benchmarks:\n");
  for (int block_size : block_sizes) {
    double elapsed = 0.0;
    for (int i = 0; i < repeat; i++) elapsed += submit_backward1(mem_dout, mem_dx, B, C, H, W, block_size, false);
    elapsed /= repeat;
    float gflops = static_cast<float>(S) / elapsed * 1e3f / 1e9f;
    std::printf("block_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed, gflops);
  }

  std::printf("Checking backward2 pass\n");
  for (int block_size : block2D_sizes) {
    std::printf("Checking block size %d\n", block_size);
    submit_backward2(mem_dout, mem_dx, B, C, H, W, block_size, true);
    ok = validate_result(gpu_dx, dx, "dx", S) && ok;
  }

  std::printf("Backward2 pass benchmarks:\n");
  for (int block_size : block2D_sizes) {
    double elapsed = 0.0;
    for (int i = 0; i < repeat; i++) elapsed += submit_backward2(mem_dout, mem_dx, B, C, H, W, block_size, false);
    elapsed /= repeat;
    float gflops = static_cast<float>(S) / elapsed * 1e3f / 1e9f;
    std::printf("block2D_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed, gflops);
  }

  std::free(x);
  std::free(out);
  std::free(dout);
  std::free(dx);
  std::free(gpu_out);
  std::free(gpu_dx);
  platform.finalize();
  return ok ? 0 : 1;
}
