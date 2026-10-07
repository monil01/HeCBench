#include <math.h>

extern "C" __global__ void axpby_kernel(
    const int total,
    const float a,
    const float b,
    const int arg_to_check,
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ z,
    int* noop)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const float xv = x[idx];
  const float yv = y[idx];
  z[idx] = a * xv + b * yv;

  bool finite = true;
  if (arg_to_check == -1) finite = isfinite(xv) && isfinite(yv);
  if (arg_to_check == 0) finite = isfinite(xv);
  if (arg_to_check == 1) finite = isfinite(yv);
  if (!finite) atomicExch(noop, 1);
}
