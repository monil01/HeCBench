#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 256

#define ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define ONE_FP16  __ushort_as_half((unsigned short)0x3c00U)
#define ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x0000U)
#define ONE_BF16  __ushort_as_bfloat16((unsigned short)0x3f80U)

extern "C" __global__ void f16_atomic_global(__half* result, int n)
{
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  __half2* result_v = reinterpret_cast<__half2*>(result);
  __half2 val {ZERO_FP16, ONE_FP16};
  atomicAdd(&result_v[tid % BLOCK_SIZE], val);
}

extern "C" __global__ void bf16_atomic_global(__nv_bfloat16* result, int n)
{
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  __nv_bfloat162* result_v = reinterpret_cast<__nv_bfloat162*>(result);
  __nv_bfloat162 val {ZERO_BF16, ONE_BF16};
  atomicAdd(&result_v[tid % BLOCK_SIZE], val);
}
