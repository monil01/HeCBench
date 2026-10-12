#include <hip/hip_runtime.h>

#include "types.h"

template <class TData>
__device__ void copy_element(TData* __restrict out, const TData* __restrict in, int num_elements) {
  const int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos < num_elements) out[pos] = in[pos];
}

#define DEFINE_COPY_KERNEL(NAME, TYPE) \
  extern "C" __global__ void NAME(unsigned char* out, const unsigned char* in, int num_elements) { \
    copy_element(reinterpret_cast<TYPE*>(out), reinterpret_cast<const TYPE*>(in), num_elements); \
  }

DEFINE_COPY_KERNEL(copy_uchar_misaligned, uchar_misaligned)
DEFINE_COPY_KERNEL(copy_uchar4_misaligned, uchar4_misaligned)
DEFINE_COPY_KERNEL(copy_uchar4_aligned, uchar4_aligned)
DEFINE_COPY_KERNEL(copy_ushort_misaligned, ushort_misaligned)
DEFINE_COPY_KERNEL(copy_uint_aligned, uint_aligned)
DEFINE_COPY_KERNEL(copy_uint2_misaligned, uint2_misaligned)
DEFINE_COPY_KERNEL(copy_uint2_aligned, uint2_aligned)
DEFINE_COPY_KERNEL(copy_uint3_misaligned, uint3_misaligned)
DEFINE_COPY_KERNEL(copy_uint3_aligned, uint3_aligned)
DEFINE_COPY_KERNEL(copy_uint4_misaligned, uint4_misaligned)
DEFINE_COPY_KERNEL(copy_uint4_aligned, uint4_aligned)
DEFINE_COPY_KERNEL(copy_uint8_misaligned, uint8_misaligned)
DEFINE_COPY_KERNEL(copy_uint8_aligned, uint8_aligned)
