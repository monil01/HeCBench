#include <float.h>

struct __attribute__((aligned(8))) Point2 {
  float x, y;
};

__device__ __forceinline__ float hd(const Point2 ap, const Point2 bp)
{
  return (ap.x - bp.x) * (ap.x - bp.x) + (ap.y - bp.y) * (ap.y - bp.y);
}

__device__ __forceinline__ void atomic_max_float(float* address, float val)
{
  unsigned int* ptr = reinterpret_cast<unsigned int*>(address);
  unsigned int ret = *ptr;
  while (val > __uint_as_float(ret)) {
    unsigned int old = ret;
    ret = atomicCAS(ptr, old, __float_as_uint(val));
    if (ret == old) break;
  }
}

extern "C" __global__ void computeDistance(
    const Point2* __restrict__ Apoints,
    const Point2* __restrict__ Bpoints,
    float* __restrict__ distance,
    const int numA,
    const int numB,
    const int distance_index)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numA) return;

  float d = FLT_MAX;
  Point2 p = Apoints[i];
  for (int j = 0; j < numB; j++) {
    float t = hd(p, Bpoints[j]);
    d = fminf(t, d);
  }

  atomic_max_float(&distance[distance_index], d);
}
