#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define BLOCK_SIZE 256

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

static float bfloat16_to_float(uint16_t h)
{
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static int launch_atomic(const char* kernel_name, iris::DMem& mem_result, int n)
{
  const size_t gws[1] = {
    ((static_cast<size_t>(n) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
  };
  const size_t lws[1] = {BLOCK_SIZE};

  iris::Task task;
  void* params[2] = {&mem_result, &n};
  int pinfo[2] = {iris_rw, sizeof(n)};
  task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws),
              const_cast<size_t*>(lws), 2, params, pinfo);
  task.flush_out(mem_result);
  return task.submit(iris_gpu, nullptr, 1);
}

static int atomic_cost(const char* kernel_name,
                       int nelems,
                       int repeat,
                       float (*to_float)(uint16_t))
{
  const size_t result_count = BLOCK_SIZE * 2;
  std::vector<uint16_t> result(result_count, 0);
  iris::DMem mem_result(result.data(), result_count * sizeof(uint16_t));
  const int n = nelems / 2;

  int rc = launch_atomic(kernel_name, mem_result, n);
  if (rc != IRIS_SUCCESS) {
    std::fprintf(stderr, "IRIS warmup task failed with code %d\n", rc);
    return 1;
  }

  std::printf("Print the first two elements in HEX: 0x%04x 0x%04x\n",
              result[0], result[1]);
  std::printf("Print the first two elements in FLOAT32: %f %f\n\n",
              to_float(result[0]), to_float(result[1]));

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    rc = launch_atomic(kernel_name, mem_result, n);
    if (rc != IRIS_SUCCESS) {
      std::fprintf(stderr, "IRIS timed task failed with code %d\n", rc);
      return 1;
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of 16-bit floating-point atomic add on global memory: %f (us)\n",
              time * 1e-3f / repeat);
  return 0;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <N> <repeat>\n", argv[0]);
    std::printf("N: total number of elements (a multiple of 2)\n");
    return 1;
  }
  const int nelems = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  assert(nelems > 0 && (nelems % 2) == 0);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::printf("\nFP16 atomic add\n");
  int rc = atomic_cost("f16_atomic_global", nelems, repeat, half_to_float);

  std::printf("\nBF16 atomic add\n");
  rc |= atomic_cost("bf16_atomic_global", nelems, repeat, bfloat16_to_float);

  std::printf("%s\n", rc == 0 ? "PASS" : "FAIL");
  platform.finalize();
  return rc;
}
