#include <hip/hip_runtime.h>
extern "C" __global__ void allocation_pattern(int* data, size_t count, int tag) {
  const size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i < count) data[i] = static_cast<int>(i % 251) + tag*257;
}
