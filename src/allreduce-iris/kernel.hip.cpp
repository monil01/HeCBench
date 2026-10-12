#include <hip/hip_runtime.h>

extern "C" __global__ void kernel_add_segment(const float* src, float* dst,
                                               size_t offset, size_t length) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < length) dst[offset + i] += src[i];
}
