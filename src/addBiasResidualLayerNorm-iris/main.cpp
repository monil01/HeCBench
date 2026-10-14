#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <random>

#include <iris/iris.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
using HostHalf = __half;
using HostBFloat16 = __hip_bfloat16;
#include "reference.h"

template <typename T> static float to_float(T value) {
  return static_cast<float>(value);
}

template <typename T> static T from_float(float value) { return static_cast<T>(value); }

static const char* kernel_for_float() { return "addBiasResidualLayerNorm_float"; }
static const char* kernel_for_half() { return "addBiasResidualLayerNorm_half"; }
static const char* kernel_for_bfloat16() { return "addBiasResidualLayerNorm_bfloat16"; }

template <typename T>
static void invoke(iris::DMem& output, iris::DMem& input, iris::DMem& bias,
                   iris::DMem& gamma, iris::DMem& beta, float epsilon, int m,
                   int n, const char* kernel_name, int version) {
  const size_t threads = version == 2 ? n / 8 : 256;
  char symbol[128];
  std::snprintf(symbol, sizeof(symbol), "%s_v%d", kernel_name, version);
  size_t global[] = {static_cast<size_t>(m) * threads};
  size_t local[] = {threads};
  void* params[] = {&output, &input, &bias, &gamma, &beta, &epsilon, &n};
  int pinfo[] = {iris_rw, iris_r, iris_r, iris_r, iris_r, sizeof(epsilon), sizeof(n)};

  iris::Task task;
  task.kernel(symbol, 1, nullptr, global, local, 7, params, pinfo);
  if (task.submit(iris_gpu, nullptr, true) != IRIS_SUCCESS) std::exit(2);
}

template <typename T>
static bool layer(int m, int n, int repeat, const char* kernel_name, int version) {
  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.f, 1.f);
  const int elements = m * n;

  T* h_input = static_cast<T*>(malloc(static_cast<size_t>(elements) * sizeof(T)));
  T* h_output = static_cast<T*>(calloc(static_cast<size_t>(elements), sizeof(T)));
  T* h_bias = static_cast<T*>(malloc(static_cast<size_t>(n) * sizeof(T)));
  T* h_gamma = static_cast<T*>(malloc(static_cast<size_t>(n) * sizeof(T)));
  T* h_beta = static_cast<T*>(malloc(static_cast<size_t>(n) * sizeof(T)));
  for (int i = 0; i < elements; ++i) h_input[i] = from_float<T>(dis(gen));
  for (int i = 0; i < n; ++i) {
    h_bias[i] = from_float<T>(dis(gen));
    h_gamma[i] = from_float<T>(dis(gen));
    h_beta[i] = from_float<T>(dis(gen));
  }

  const float epsilon = 1e-6f;
  iris::DMem mem_input(h_input, static_cast<size_t>(elements) * sizeof(T));
  iris::DMem mem_output(h_output, static_cast<size_t>(elements) * sizeof(T));
  iris::DMem mem_bias(h_bias, static_cast<size_t>(n) * sizeof(T));
  iris::DMem mem_gamma(h_gamma, static_cast<size_t>(n) * sizeof(T));
  iris::DMem mem_beta(h_beta, static_cast<size_t>(n) * sizeof(T));

  T* reference_output = static_cast<T*>(calloc(static_cast<size_t>(elements), sizeof(T)));
  for (int i = 0; i < 100; ++i) {
    invoke<T>(mem_output, mem_input, mem_bias, mem_gamma, mem_beta, epsilon, m, n,
              kernel_name, version);
    reference<T>(reference_output, h_input, h_bias, h_gamma, h_beta, epsilon, m, n);
  }
  iris::Task flush;
  flush.flush_out(mem_output);
  if (flush.submit(iris_gpu, nullptr, true) != IRIS_SUCCESS) std::exit(2);

  bool ok = true;
  const float error_bound = sizeof(T) >= 4 ? 1e-4f : 0.5f;
  for (int i = 0; i < elements; ++i) {
    if (!std::isfinite(to_float(h_output[i])) || std::fabs(to_float(h_output[i]) - to_float(reference_output[i])) > error_bound) {
      std::printf("i=%d %f != %f (ref)\n", i, to_float(h_output[i]),
                  to_float(reference_output[i]));
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  free(reference_output);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i)
    invoke<T>(mem_output, mem_input, mem_bias, mem_gamma, mem_beta, epsilon, m, n,
              kernel_name, version);
  auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of AddBiasResidualLayerNorm (%d x %d): %f (us)\n",
              m, n, (elapsed * 1e-3f) / repeat);

  free(h_input);
  free(h_output);
  free(h_bias);
  free(h_gamma);
  free(h_beta);
  return ok;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const char* rows = std::getenv("HECBENCH_VALIDATE_ROWS");
  const int m = rows ? std::atoi(rows) : 4096;
  if (m <= 0 || repeat <= 0) return 1;
  bool ok = true;
  for (int n = 256; n <= 8192; n *= 2) {
    std::printf("---------------- float32 (version 1) -------------\n");
    ok = layer<float>(m, n, repeat, kernel_for_float(), 1) && ok;
    std::printf("---------------- float32 (version 2) -------------\n");
    ok = layer<float>(m, n, repeat, kernel_for_float(), 2) && ok;
    std::printf("---------------- float16 (version 1) -------------\n");
    ok = layer<HostHalf>(m, n, repeat, kernel_for_half(), 1) && ok;
    std::printf("---------------- float16 (version 2) -------------\n");
    ok = layer<HostHalf>(m, n, repeat, kernel_for_half(), 2) && ok;
    std::printf("---------------- bfloat16 (version 1) -------------\n");
    ok = layer<HostBFloat16>(m, n, repeat, kernel_for_bfloat16(), 1) && ok;
    std::printf("---------------- bfloat16 (version 2) -------------\n");
    ok = layer<HostBFloat16>(m, n, repeat, kernel_for_bfloat16(), 2) && ok;
  }
  platform.finalize();
  return ok ? 0 : 1;
}
