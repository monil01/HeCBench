#include <hip/hip_runtime.h>

extern "C" __global__ void floydWarshallPass(unsigned int* __restrict__ pathDistanceBuffer,
                                             unsigned int* __restrict__ pathBuffer,
                                             const unsigned int numNodes,
                                             const unsigned int pass)
{
  const unsigned int xValue = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int yValue = threadIdx.y + blockIdx.y * blockDim.y;
  if (xValue >= numNodes || yValue >= numNodes) return;

  const unsigned int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue];
  const unsigned int tempWeight = pathDistanceBuffer[yValue * numNodes + pass] +
                                  pathDistanceBuffer[pass * numNodes + xValue];
  if (tempWeight < oldWeight) {
    pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight;
    pathBuffer[yValue * numNodes + xValue] = pass;
  }
}
