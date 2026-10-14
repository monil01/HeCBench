#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "fresnel_compat.h"
#include "fresnel_sources.h"

#define GPU_NUM_THREADS 256

static void reference(const double* input, double* output, int n)
{
  for (int i = 0; i < n; i++) {
    output[i] = Fresnel_Sine_Integral(input[i]);
  }
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
  const char* bound = std::getenv("HECBENCH_FRESNEL_POINTS");
  const int points = bound ? std::atoi(bound) : static_cast<int>(8.0 / 1e-7);
  if (points < 2) { platform.finalize(); return 1; }
  const double interval = bound ? 8.0 / (points - 1) : 1e-7;
  const size_t points_size = static_cast<size_t>(points) * sizeof(double);

  double* x = static_cast<double*>(std::malloc(points_size));
  double* output = static_cast<double*>(std::malloc(points_size));
  double* h_output = static_cast<double*>(std::malloc(points_size));
  if (!x || !output || !h_output) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(x);
    std::free(output);
    std::free(h_output);
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < points; i++) {
    x[i] = static_cast<double>(i) * interval;
  }

  iris::DMem mem_x(x, points_size);
  iris::DMem mem_output(output, points_size);

  const size_t gws[1] = { static_cast<size_t>((points + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS) * GPU_NUM_THREADS };
  const size_t lws[1] = { GPU_NUM_THREADS };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = { &mem_x, &mem_output, const_cast<int*>(&points) };
    int pinfo[3] = { iris_r, iris_w, sizeof(points) };
    task.kernel("fresnel_kernel", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 3, params, pinfo);
    task.flush_out(mem_output);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  if (const char* path = std::getenv("HECBENCH_FRESNEL_DUMP")) {
    FILE* file = std::fopen(path, "wb");
    if (!file || std::fwrite(output, sizeof(double), points, file) != static_cast<size_t>(points)) {
      if (file) std::fclose(file);
      std::fprintf(stderr, "Fresnel verification dump failed\n");
      platform.finalize();
      return 1;
    }
    std::fclose(file);
  }
  if (std::getenv("HECBENCH_FRESNEL_CAPTURE_ONLY")) {
    std::printf("GPU output captured; numerical verification deferred to external verifier\n");
    std::free(x); std::free(output); std::free(h_output);
    platform.finalize();
    return 0;
  }
  reference(x, h_output, points);
  bool ok = true;
  for (int i = 0; i < points; i++) {
    if (!std::isfinite(h_output[i]) || !std::isfinite(output[i]) ||
        std::fabs(h_output[i] - output[i]) > 1e-6) {
      std::printf("%lf %lf\n", h_output[i], output[i]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(x);
  std::free(output);
  std::free(h_output);
  platform.finalize();
  return ok ? 0 : 1;
}
