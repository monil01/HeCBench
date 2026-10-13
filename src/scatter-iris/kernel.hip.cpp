#include <hip/hip_runtime.h>
#include <cstdint>
#include "../scatter-cuda/reducer.h"

// The source uses B=K=1 and a contiguous one-dimensional index tensor.
// Flattening that descriptor exposes its pointer as a dmem argument.
#define KERNEL(TYPE, TAG, OP, NAME) \
extern "C" __global__ void scatter_##TAG##_##NAME(const TYPE* src, const int64_t* idx, TYPE* out, int numel) { \
  int i = blockIdx.x * blockDim.x + threadIdx.x; \
  if (i < numel) Reducer<TYPE, OP>::atomic_write(out + idx[i], src[i]); \
}
#define CASES(TYPE, TAG) \
KERNEL(TYPE, TAG, SUM, sum) \
KERNEL(TYPE, TAG, MUL, mul) \
KERNEL(TYPE, TAG, DIV, div) \
KERNEL(TYPE, TAG, MIN, min) \
KERNEL(TYPE, TAG, MAX, max)
CASES(int32_t, i32)
CASES(int64_t, i64)
CASES(float, f32)
CASES(double, f64)
