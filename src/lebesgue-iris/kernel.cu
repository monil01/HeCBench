#include <math.h>
#include <cuda.h>

__device__ __forceinline__
double atomicMax(double* address, double val)
{
  unsigned long long* ptr = (unsigned long long*)address;
  unsigned long long ret = *ptr;
  while (val > __longlong_as_double(ret)) {
    unsigned long long old = ret;
    if ((ret = atomicCAS(ptr, old, __double_as_longlong(val))) == old)
      break;
  }
  return __longlong_as_double(ret);
}

extern "C" __global__
void lebesgue_kernel(double* __restrict__ lmax,
                     double* __restrict__ linterp,
                     const double* __restrict__ xfun,
                     const double* __restrict__ x,
                     const int n, const int nfun)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nfun) return;

  double t = 0.0;
  for (int i1 = 0; i1 < n; i1++) {
    linterp[i1 * nfun + j] = 1.0;
    for (int i2 = 0; i2 < n; i2++)
      if (i1 != i2)
        linterp[i1 * nfun + j] =
          linterp[i1 * nfun + j] * (xfun[j] - x[i2]) / (x[i1] - x[i2]);
    t += fabs(linterp[i1 * nfun + j]);
  }

  atomicMax(lmax, t);
}
