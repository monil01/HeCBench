#include <cuda_runtime.h>
extern "C" __global__ void triad(const float*__restrict__ A,
                      const float*__restrict__ B,
                            float*__restrict__ C,
                            float s)
{
  int gid = threadIdx.x + (blockIdx.x * blockDim.x);
  C[gid] = A[gid] + s*B[gid];
}
