#include <hip/hip_runtime.h>

#define VECTOR_SIZE 1024

extern "C" __global__ void benchmark_constant_i32(int* output,
                                                   const int* input,
                                                   int repeat)
{
  (void)repeat;
  int sum = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < VECTOR_SIZE; j += 4) {
      sum += input[j + i];
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = sum;
  }
}

extern "C" __global__ void benchmark_constant_i32x2(int* output,
                                                     const int* input,
                                                     int repeat)
{
  (void)repeat;
  const int2* data = reinterpret_cast<const int2*>(input);
  int2 sum = make_int2(0, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < VECTOR_SIZE / 2; j += 4) {
      int2 v = data[j + i];
      sum.x += v.x;
      sum.y += v.y;
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = sum.x + sum.y;
  }
}

extern "C" __global__ void benchmark_constant_i32x4(int* output,
                                                     const int* input,
                                                     int repeat)
{
  (void)repeat;
  const int4* data = reinterpret_cast<const int4*>(input);
  int4 sum = make_int4(0, 0, 0, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < VECTOR_SIZE / 4; j += 4) {
      int4 v = data[j + i];
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = sum.x + sum.y + sum.z + sum.w;
  }
}
