#include <hip/hip_runtime.h>
extern "C" __global__ void
fan1 (const float*__restrict__ a,
            float*__restrict__ m,
      const int size, const int t)
{
  int globalId = blockDim.x * blockIdx.x + threadIdx.x;
  if (globalId < size-1-t) {
    m[size * (globalId + t + 1)+t] = 
      a[size * (globalId + t + 1) + t] / a[size * t + t];
  }
}

extern "C" __global__ void
fan2 (float*__restrict__ a,
      float*__restrict__ b,
      const float*__restrict__ m,
      const int size, const int t)
{
  int globalIdy = blockDim.x * blockIdx.x + threadIdx.x;
  int globalIdx = blockDim.y * blockIdx.y + threadIdx.y;
  if (globalIdx < size-1-t && globalIdy < size-t) {
    a[size*(globalIdx+1+t)+(globalIdy+t)] -= 
      m[size*(globalIdx+1+t)+t] * a[size*t+(globalIdy+t)];

    if(globalIdy == 0){
      b[globalIdx+1+t] -= 
        m[size*(globalIdx+1+t)+(globalIdy+t)] * b[t];
    }
  }
}
