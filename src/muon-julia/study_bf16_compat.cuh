#include <cuda_runtime.h>
#include <cuda_bf16.h>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
__device__ __forceinline__ float2 study_pair_to_float(const __nv_bfloat162 a) { return make_float2(__bfloat162float(a.x), __bfloat162float(a.y)); }
__device__ __forceinline__ __nv_bfloat162 study_float_to_pair(const float2 a) { __nv_bfloat162 out; out.x=__float2bfloat16(a.x); out.y=__float2bfloat16(a.y); return out; }
#define __bfloat1622float2 study_pair_to_float
#define __float22bfloat162_rn study_float_to_pair
#endif
