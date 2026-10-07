#include <hip/hip_runtime.h>

extern "C" __global__ void atomicAggregate_k(int* d, int s)
{
  const int t = threadIdx.x;
  int* ptr = d + (t % s);
  atomicAdd(ptr, 1);
}
