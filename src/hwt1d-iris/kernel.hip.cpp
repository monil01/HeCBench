#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__ void dwtHaar1D(
    const float* __restrict__ inSignal,
    float* __restrict__ coefsSignal,
    float* __restrict__ AverageSignal,
    unsigned int tLevels,
    unsigned int signalLength,
    unsigned int levelsDone,
    unsigned int mLevels)
{
  size_t localId = threadIdx.x;
  size_t groupId = blockIdx.x;
  size_t localSize = blockDim.x;
  __shared__ float sharedArray[512];

  float t0 = inSignal[groupId * localSize * 2 + localId];
  float t1 = inSignal[groupId * localSize * 2 + localSize + localId];
  if (levelsDone == 0) {
    float r = rsqrtf((float)signalLength);
    t0 *= r;
    t1 *= r;
  }
  sharedArray[localId] = t0;
  sharedArray[localSize + localId] = t1;
  __syncthreads();

  unsigned int levels = tLevels > mLevels ? mLevels : tLevels;
  unsigned int activeThreads = (1 << levels) / 2;
  unsigned int midOutPos = signalLength / 2;
  const float rsqrt_two = 0.7071f;

  for (unsigned int i = 0; i < levels; ++i) {
    float data0 = 0.0f;
    float data1 = 0.0f;
    if (localId < activeThreads) {
      data0 = sharedArray[2 * localId];
      data1 = sharedArray[2 * localId + 1];
    }
    __syncthreads();

    if (localId < activeThreads) {
      sharedArray[localId] = (data0 + data1) * rsqrt_two;
      unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
      coefsSignal[globalPos] = (data0 - data1) * rsqrt_two;
      midOutPos >>= 1;
    }
    activeThreads >>= 1;
    __syncthreads();
  }

  if (localId == 0) {
    AverageSignal[groupId] = sharedArray[0];
  }
}
