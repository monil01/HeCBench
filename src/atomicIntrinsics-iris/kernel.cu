#include <cuda_runtime.h>

template <class T>
__device__ void test_kernel_impl(T *g_odata, size_t len)
{
  const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= len) return;

  atomicAdd(&g_odata[0], (T)10);
  atomicSub(&g_odata[1], (T)10);
  atomicMax(&g_odata[2], (T)tid);
  atomicMin(&g_odata[3], (T)tid);
  atomicAnd(&g_odata[4], (T)(2*tid+7));
  atomicOr(&g_odata[5], (T)(1 << tid));
  atomicXor(&g_odata[6], (T)tid);
  atomicInc((unsigned int*)&g_odata[7], 17);
  atomicDec((unsigned int*)&g_odata[8], 137);
}

extern "C" __global__ void testKernel_int(int *g_odata, size_t len)
{
  test_kernel_impl(g_odata, len);
}

extern "C" __global__ void testKernel_uint(unsigned int *g_odata, size_t len)
{
  test_kernel_impl(g_odata, len);
}
