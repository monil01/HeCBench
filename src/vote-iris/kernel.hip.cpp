#include <hip/hip_runtime.h>

extern "C" __global__ void VoteAnyKernel1(const unsigned int* input,
                                           unsigned int* result,
                                           int repeat)
{
  int tx = threadIdx.x;
  for (int i = 0; i < repeat; i++) result[tx] = __any(input[tx]);
}

extern "C" __global__ void VoteAllKernel2(const unsigned int* input,
                                           unsigned int* result,
                                           int repeat)
{
  int tx = threadIdx.x;
  for (int i = 0; i < repeat; i++) result[tx] = __all(input[tx]);
}

extern "C" __global__ void VoteAnyKernel3(bool* info, int warp_size, int repeat)
{
  int tx = threadIdx.x;
  for (int i = 0; i < repeat; i++) {
    bool* offs = info + (tx * 3);
    *offs = __any((tx >= (warp_size * 3) / 2));
    *(offs + 1) = (tx >= (warp_size * 3) / 2 ? true : false);
    if (__all((tx >= (warp_size * 3) / 2))) *(offs + 2) = true;
  }
}

extern "C" __global__ void VoteResetInfo(bool* info, int count)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx < count) info[tx] = false;
}
