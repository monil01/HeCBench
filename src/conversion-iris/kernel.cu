#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using uchar = unsigned char;

template <typename Td, typename Ts>
__device__ void cvt_one(Td* __restrict__ dst, const Ts* __restrict__ src, int nelems)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = static_cast<Td>(src[i]);
  }
}

#define DEFINE_CVT(SRC_NAME, DST_NAME, SRC_TYPE, DST_TYPE) \
  extern "C" __global__ void cvt_##SRC_NAME##_##DST_NAME( \
      DST_TYPE* __restrict__ dst, const SRC_TYPE* __restrict__ src, int nelems) \
  { \
    cvt_one<DST_TYPE, SRC_TYPE>(dst, src, nelems); \
  }

DEFINE_CVT(bfloat16, half,  __nv_bfloat16, half)
DEFINE_CVT(bfloat16, float, __nv_bfloat16, float)
DEFINE_CVT(bfloat16, int,   __nv_bfloat16, int)
DEFINE_CVT(bfloat16, char,  __nv_bfloat16, char)
DEFINE_CVT(bfloat16, uchar, __nv_bfloat16, uchar)

DEFINE_CVT(half, half,  half, half)
DEFINE_CVT(half, float, half, float)
DEFINE_CVT(half, int,   half, int)
DEFINE_CVT(half, char,  half, char)
DEFINE_CVT(half, uchar, half, uchar)

DEFINE_CVT(float, float, float, float)
DEFINE_CVT(float, half,  float, half)
DEFINE_CVT(float, int,   float, int)
DEFINE_CVT(float, char,  float, char)
DEFINE_CVT(float, uchar, float, uchar)

DEFINE_CVT(int, int,   int, int)
DEFINE_CVT(int, float, int, float)
DEFINE_CVT(int, half,  int, half)
DEFINE_CVT(int, char,  int, char)
DEFINE_CVT(int, uchar, int, uchar)

DEFINE_CVT(char, int,   char, int)
DEFINE_CVT(char, float, char, float)
DEFINE_CVT(char, half,  char, half)
DEFINE_CVT(char, char,  char, char)
DEFINE_CVT(char, uchar, char, uchar)

DEFINE_CVT(uchar, int,   uchar, int)
DEFINE_CVT(uchar, float, uchar, float)
DEFINE_CVT(uchar, half,  uchar, half)
DEFINE_CVT(uchar, char,  uchar, char)
DEFINE_CVT(uchar, uchar, uchar, uchar)
