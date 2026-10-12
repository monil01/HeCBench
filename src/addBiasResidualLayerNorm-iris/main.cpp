#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <random>

#include <iris/iris.hpp>

using HostHalf = _Float16;

struct HostBFloat16 {
  uint16_t bits;
};

static_assert(sizeof(HostHalf) == 2, "host half must match HIP half");
static_assert(sizeof(HostBFloat16) == 2, "host bfloat16 must match HIP bfloat16");

#include "reference.h"

template <typename T> static float to_float(T value) {
  return static_cast<float>(value);
}

template <> float to_float<HostBFloat16>(HostBFloat16 value) {
  uint32_t bits = static_cast<uint32_t>(value.bits) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template <typename T> static T from_float(float value) {
  return static_cast<T>(value);
}

template <> HostBFloat16 from_float<HostBFloat16>(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return HostBFloat16{static_cast<uint16_t>(bits >> 16)};
}

static const char* kernel_for_float() { return "addBiasResidualLayerNorm_float"; }
static const char* kernel_for_half() { return "addBiasResidualLayerNorm_half"; }
static const char* kernel_for_bfloat16() { return "addBiasResidualLayerNorm_bfloat16"; }

template <typename T>
static void invoke(iris::DMem& output, iris::DMem& input, iris::DMem& bias,
                   iris::DMem& gamma, iris::DMem& beta, float epsilon, int m,
                   int n, const char* kernel_name) {
  constexpr size_t threads = 256;
  size_t global[] = {static_cast<size_t>(m) * threads};
  size_t local[] = {threads};
  void* params[] = {&output, &input, &bias, &gamma, &beta, &epsilon, &n};
  int pinfo[] = {iris_rw, iris_r, iris_r, iris_r, iris_r, sizeof(epsilon), sizeof(n)};

  iris::Task task;
  task.kernel(kernel_name, 1, nullptr, global, local, 7, params, pinfo);
  task.submit(iris_gpu, nullptr, true);
}

template <typename T>
static void layer(int m, int n, int repeat, const char* kernel_name) {
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
              kernel_name);
    reference<T>(reference_output, h_input, h_bias, h_gamma, h_beta, epsilon, m, n);
  }
  mem_output.fetch(h_output);

  bool ok = true;
  const float error_bound = sizeof(T) >= 4 ? 1e-4f : 0.5f;
  for (int i = 0; i < elements; ++i) {
    if (std::fabs(to_float(h_output[i]) - to_float(reference_output[i])) > error_bound) {
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
              kernel_name);
  auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of AddBiasResidualLayerNorm (%d x %d): %f (us)\n",
              m, n, (elapsed * 1e-3f) / repeat);

  free(h_input);
  free(h_output);
  free(h_bias);
  free(h_gamma);
  free(h_beta);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int m = 4096;
  for (int n = 256; n <= 8192; n *= 2) {
    std::printf("---------------- float32 (version 1) -------------\n");
    layer<float>(m, n, repeat, kernel_for_float());
    std::printf("---------------- float32 (version 2) -------------\n");
    layer<float>(m, n, repeat, kernel_for_float());
    std::printf("---------------- float16 (version 1) -------------\n");
    layer<HostHalf>(m, n, repeat, kernel_for_half());
    std::printf("---------------- float16 (version 2) -------------\n");
    layer<HostHalf>(m, n, repeat, kernel_for_half());
    std::printf("---------------- bfloat16 (version 1) -------------\n");
    layer<HostBFloat16>(m, n, repeat, kernel_for_bfloat16());
    std::printf("---------------- bfloat16 (version 2) -------------\n");
    layer<HostBFloat16>(m, n, repeat, kernel_for_bfloat16());
  }
  platform.finalize();
  return 0;
}
