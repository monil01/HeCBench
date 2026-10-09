#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#define BLOCK_SIZE 256

static uint16_t float_to_bfloat16(float value)
{
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

static float bfloat16_to_float(uint16_t value)
{
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static uint16_t float_to_half(float value)
{
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;

  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t half_mant = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) ++half_mant;
    return static_cast<uint16_t>(sign | half_mant);
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);

  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  if (mant & 0x00001000u) ++half;
  return static_cast<uint16_t>(half);
}

static float half_to_float(uint16_t h)
{
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03ffu;
      bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename T>
static double to_double(T value)
{
  return static_cast<double>(value);
}

template <>
double to_double<uint16_t>(uint16_t value)
{
  return half_to_float(value);
}

static double to_bfloat_double(uint16_t value)
{
  return bfloat16_to_float(value);
}

template <typename T>
static void fill_inputs(std::vector<T>& srcA, std::vector<T>& srcB, double& sum)
{
  const float value = static_cast<float>(std::sqrt(65504.0 / srcA.size()));
  sum = 0.0;
  for (size_t i = 0; i < srcA.size(); ++i) {
    srcA[i] = static_cast<T>(value);
    srcB[i] = static_cast<T>(value);
    sum += static_cast<double>(static_cast<float>(srcA[i])) *
           static_cast<double>(static_cast<float>(srcB[i]));
  }
}

static void fill_half_inputs(std::vector<uint16_t>& srcA,
                             std::vector<uint16_t>& srcB,
                             double& sum,
                             bool bfloat)
{
  const float value = static_cast<float>(std::sqrt(65504.0 / srcA.size()));
  sum = 0.0;
  for (size_t i = 0; i < srcA.size(); ++i) {
    srcA[i] = bfloat ? float_to_bfloat16(value) : float_to_half(value);
    srcB[i] = srcA[i];
    const double converted = bfloat ? to_bfloat_double(srcA[i]) : to_double(srcA[i]);
    sum += converted * converted;
  }
}

static int run_dot_kernel(const char* partial_kernel,
                          const char* final_kernel,
                          iris::DMem& mem_a,
                          iris::DMem& mem_b,
                          iris::DMem& mem_partial,
                          iris::DMem& mem_dst,
                          size_t n,
                          int num_blocks)
{
  const size_t gws_partial[1] = {static_cast<size_t>(num_blocks) * BLOCK_SIZE};
  const size_t lws[1] = {BLOCK_SIZE};
  int n_arg = static_cast<int>(n);

  iris::Task partial_task;
  void* partial_params[4] = {&mem_partial, &mem_a, &mem_b, &n_arg};
  int partial_pinfo[4] = {iris_w, iris_r, iris_r, sizeof(n_arg)};
  partial_task.kernel(partial_kernel, 1, nullptr, const_cast<size_t*>(gws_partial),
                      const_cast<size_t*>(lws), 4, partial_params, partial_pinfo);
  int rc = partial_task.submit(iris_gpu, nullptr, 1);
  if (rc != IRIS_SUCCESS) return rc;

  iris::Task final_task;
  void* final_params[3] = {&mem_dst, &mem_partial, &num_blocks};
  int final_pinfo[3] = {iris_w, iris_r, sizeof(num_blocks)};
  final_task.kernel(final_kernel, 1, nullptr, const_cast<size_t*>(lws),
                    const_cast<size_t*>(lws), 3, final_params, final_pinfo);
  final_task.flush_out(mem_dst);
  return final_task.submit(iris_gpu, nullptr, 1);
}

template <typename T>
static int dot_numeric(const char* label,
                       const char* partial_kernel,
                       size_t elements,
                       int iterations)
{
  const int num_blocks = static_cast<int>((elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
  std::vector<T> srcA(elements);
  std::vector<T> srcB(elements);
  std::vector<double> partial(num_blocks, 0.0);
  double dst = 0.0;
  double sum = 0.0;
  fill_inputs(srcA, srcB, sum);

  iris::DMem mem_a(srcA.data(), srcA.size() * sizeof(T));
  iris::DMem mem_b(srcB.data(), srcB.size() * sizeof(T));
  iris::DMem mem_partial(partial.data(), partial.size() * sizeof(double));
  iris::DMem mem_dst(&dst, sizeof(dst));

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    const int rc = run_dot_kernel(partial_kernel, "dot_final_sum", mem_a, mem_b,
                                  mem_partial, mem_dst, elements, num_blocks);
    if (rc != IRIS_SUCCESS) {
      std::fprintf(stderr, "IRIS dot task failed with code %d\n", rc);
      return 1;
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average iris %s dot execution time %f (ms)\n", label,
              (time * 1e-6f) / iterations);
  std::printf("Host: %lf  Device: %lf\n", sum, dst);
  const bool pass = std::fabs(dst - sum) < 1e-1;
  std::printf("%s\n\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

static int dot_half_like(const char* label,
                         const char* partial_kernel,
                         size_t elements,
                         int iterations,
                         bool bfloat)
{
  const int num_blocks = static_cast<int>((elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
  std::vector<uint16_t> srcA(elements);
  std::vector<uint16_t> srcB(elements);
  std::vector<double> partial(num_blocks, 0.0);
  double dst = 0.0;
  double sum = 0.0;
  fill_half_inputs(srcA, srcB, sum, bfloat);

  iris::DMem mem_a(srcA.data(), srcA.size() * sizeof(uint16_t));
  iris::DMem mem_b(srcB.data(), srcB.size() * sizeof(uint16_t));
  iris::DMem mem_partial(partial.data(), partial.size() * sizeof(double));
  iris::DMem mem_dst(&dst, sizeof(dst));

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    const int rc = run_dot_kernel(partial_kernel, "dot_final_sum", mem_a, mem_b,
                                  mem_partial, mem_dst, elements, num_blocks);
    if (rc != IRIS_SUCCESS) {
      std::fprintf(stderr, "IRIS dot task failed with code %d\n", rc);
      return 1;
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average iris %s dot execution time %f (ms)\n", label,
              (time * 1e-6f) / iterations);
  std::printf("Host: %lf  Device: %lf\n", sum, dst);
  const bool pass = std::fabs(dst - sum) < 1e-1;
  std::printf("%s\n\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t elements = static_cast<size_t>(std::atol(argv[1]));
  const int iterations = std::atoi(argv[2]);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int rc = 0;
  std::printf("\nFP64 Dot\n");
  rc |= dot_numeric<double>("FP64", "dot_partial_f64", elements, iterations);
  std::printf("\nFP32 Dot\n");
  rc |= dot_numeric<float>("FP32", "dot_partial_f32", elements, iterations);
  std::printf("\nFP16 Dot\n");
  rc |= dot_half_like("FP16", "dot_partial_f16", elements, iterations, false);
  std::printf("\nBF16 Dot\n");
  rc |= dot_half_like("BF16", "dot_partial_bf16", elements, iterations, true);

  platform.finalize();
  return rc;
}
