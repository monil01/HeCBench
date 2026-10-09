#include <hip/hip_runtime.h>

#include <assert.h>

extern "C" __global__ void testKernel(const int N)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  assert(gid < N);
}

extern "C" __global__ void perfKernel()
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  assert(gid <= blockDim.x * gridDim.x);
  int s = 0;
  for (int n = 1; n <= gid; n++) {
    s++;
    assert(s <= gid);
  }
}

extern "C" __global__ void perfKernel2()
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int s = 0;
  for (int n = 1; n <= gid; n++) {
    s++;
    assert(s <= gid);
  }
}
