#include <cuda_runtime.h>
#include <math.h>
#include "qrg.h"
__device__
float MoroInvCNDgpu(unsigned int x)
{
  const float a1 = 2.50662823884f;
  const float a2 = -18.61500062529f;
  const float a3 = 41.39119773534f;
  const float a4 = -25.44106049637f;
  const float b1 = -8.4735109309f;
  const float b2 = 23.08336743743f;
  const float b3 = -21.06224101826f;
  const float b4 = 3.13082909833f;
  const float c1 = 0.337475482272615f;
  const float c2 = 0.976169019091719f;
  const float c3 = 0.160797971491821f;
  const float c4 = 2.76438810333863E-02f;
  const float c5 = 3.8405729373609E-03f;
  const float c6 = 3.951896511919E-04f;
  const float c7 = 3.21767881768E-05f;
  const float c8 = 2.888167364E-07f;
  const float c9 = 3.960315187E-07f;

  float z;

  bool negate = false;

  // Ensure the conversion to floating point will give a value in the
  // range (0,0.5] by restricting the input to the bottom half of the
  // input domain. We will later reflect the result if the input was
  // originally in the top half of the input domain
  if (x >= 0x80000000UL)
  {
    x = 0xffffffffUL - x;
    negate = true;
  }

  // x is now in the range [0,0x80000000) (i.e. [0,0x7fffffff])
  // Convert to floating point in (0,0.5]
  const float x1 = 1.0f / (float)0xffffffffUL;
  const float x2 = x1 / 2.0f;
  float p1 = x * x1 + x2;
  // Convert to floating point in (-0.5,0]
  float p2 = p1 - 0.5f;

  // The input to the Moro inversion is p2 which is in the range
  // (-0.5,0]. This means that our output will be the negative side
  // of the bell curve (which we will reflect if "negate" is true).

  // Main body of the bell curve for |p| < 0.42
  if (p2 > -0.42f)
  {
    z = p2 * p2;
    z = p2 * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
  }
  // Special case (Chebychev) for tail
  else
  {
    z = logf(-logf(p1));
    z = - (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))));
  }

  // If the original input (x) was in the top half of the range, reflect
  // to get the positive side of the bell curve
  return negate ? -z : z;
}

// size of output random array
const unsigned int N = 1048576;

extern "C" __global__ void
qrng (float* output, const unsigned int* table, const unsigned int seed, const unsigned int N)
{
  unsigned int globalID_x   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int localID_y    = threadIdx.y;
  unsigned int globalSize_x = gridDim.x * blockDim.x;

  for (unsigned int pos = globalID_x; pos < N; pos += globalSize_x) {
    unsigned int result = 0;
    unsigned int data = seed + pos;
    for(int bit = 0; bit < QRNG_RESOLUTION; bit++, data >>= 1)
      if(data & 1) result ^= table[bit+localID_y*QRNG_RESOLUTION];
    output[__mul24(localID_y,N) + pos] = (float)(result + 1) * INT_SCALE;
  }
}

extern "C" __global__ void
icnd (float* output, const unsigned int pathN, const unsigned int distance)
{
  const unsigned int globalID   = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int globalSize = gridDim.x * blockDim.x;

  for(unsigned int pos = globalID; pos < pathN; pos += globalSize){
    unsigned int d = (pos + 1) * distance;
    output[pos] = MoroInvCNDgpu(d);
  }
}
