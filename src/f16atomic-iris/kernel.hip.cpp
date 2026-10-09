#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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
  unsafeAtomicAdd(&result_v[tid % BLOCK_SIZE], val);
}

extern "C" __global__ void bf16_atomic_global(__hip_bfloat16* result, int n)
{
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  __hip_bfloat162* result_v = reinterpret_cast<__hip_bfloat162*>(result);
  __hip_bfloat162 val {ZERO_BF16, ONE_BF16};
  unsafeAtomicAdd(&result_v[tid % BLOCK_SIZE], val);
}
