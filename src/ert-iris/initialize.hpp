#include <inttypes.h>
#include <type_traits>
#include <typeinfo>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "rep.h"

int gpu_blocks;
int gpu_threads;


#define KERNEL1(a, b, c) ((a) = (b) + (c))
#define KERNEL2(a, b, c) ((a) = (a) * (b) + (c))
#define KERNEL2HALF(a, b, c) ((a) = __hadd2((b), (c)))
#define KERNEL4HALF(a, b, c) ((a) = __hfma2((a), (b), (c)))


// If data type is "half2"
template <typename T, typename std::enable_if<std::is_same<T, half2>::value, int>::type = 0>
void initialize(uint64_t nsize, T *__restrict__ A, float value)
{
  uint64_t i;
  for (i = 0; i < nsize; ++i) {
    A[i] = __float2half2_rn(value);
  }
}

// If data type is float or double
template <typename T, typename std::enable_if<!std::is_same<T, half2>::value, int>::type = 0>
void initialize(uint64_t nsize, T *__restrict__ A, float value)
{
  uint64_t i;
  for (i = 0; i < nsize; ++i) {
    A[i] = (T)value;
  }
}
