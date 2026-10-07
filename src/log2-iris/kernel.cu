#include <stdint.h>

union FloatBits {
  float f;
  uint32_t i;
};

__device__ float binary_log_device(float input, int precision)
{
  FloatBits d1;
  d1.f = input;
  const uint8_t exponent = static_cast<uint8_t>(((d1.i & 0x7F800000) >> 23) - 127);
  int m = 0;
  int sum_m = 0;
  float result = 0.0f;
  const int test = (1 << exponent);
  float y = input / test;
  bool max_condition_met = false;
  const uint64_t one = 1;
  uint64_t denom = 0;
  uint64_t prev_denom = 0;
  while ((sum_m < precision + 1 && y != 1.0f) || max_condition_met) {
    m = 0;
    while ((y < 2.0f) && (sum_m + m < precision + 1)) {
      y *= y;
      m++;
    }

    sum_m += m;
    prev_denom = denom;
    denom = one << sum_m;

    if (sum_m >= precision) break;
    if (prev_denom > denom) {
      max_condition_met = true;
      break;
    }

    result += 1.0f / static_cast<float>(denom);
    y /= 2.0f;
  }
  return exponent + result;
}

extern "C" __global__ void compute_log(
    float* __restrict__ output,
    const float* __restrict__ input,
    int r,
    int num_inputs,
    int precision)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_inputs) {
    output[r * num_inputs + i] = binary_log_device(input[i], precision);
  }
}
