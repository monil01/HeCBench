#include <cuda_fp16.h>
#define BLOCK_SIZE 256
#define ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define ONE_FP16 __ushort_as_half((unsigned short)0x3c00U)
__global__
void f16AtomicOnGlobalMem(__half* result, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  __half2 *result_v = reinterpret_cast<__half2*>(result);
  __half2 val {ZERO_FP16, ONE_FP16};
  atomicAdd(&result_v[tid % BLOCK_SIZE], val);
}

