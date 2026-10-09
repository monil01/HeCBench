#include <hip/hip_runtime.h>

#define TREE_NUM 4096
#define TREE_SIZE 4096

extern "C" __global__ void AoSKernel(const int* __restrict__ trees,
                                     int* __restrict__ outBuf,
                                     const int treeNumber) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= static_cast<unsigned int>(treeNumber)) return;
  unsigned int res = 0;
  for (int i = 0; i < TREE_SIZE; i++) {
    res += trees[gid * TREE_SIZE + i];
  }
  outBuf[gid] = res;
}

extern "C" __global__ void SoAKernel(const int* __restrict__ applesOnTrees,
                                     int* __restrict__ outBuf,
                                     const int treeNumber) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= static_cast<unsigned int>(treeNumber)) return;
  unsigned int res = 0;
  for (int i = 0; i < TREE_SIZE; i++) {
    res += applesOnTrees[i * TREE_NUM + gid];
  }
  outBuf[gid] = res;
}
