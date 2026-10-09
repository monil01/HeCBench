#include <hip/hip_runtime.h>

extern "C" __global__ void k0(const float* __restrict__ a,
                               float* __restrict__ o, int n)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  float x = a[t];
  o[t] = coshf(x) / sinhf(x) - 1.f / x;
}

extern "C" __global__ void k1(const float* __restrict__ a,
                               float* __restrict__ o, int n)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  float x = a[t];
  o[t] = 1.f / tanhf(x) - 1.f / x;
}

extern "C" __global__ void k2(const float* __restrict__ a,
                               float* __restrict__ o, int n)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  float x = a[t];
  float s = x * x;
  float r = 7.70960469e-8f;
  r = fmaf(r, s, -1.65101926e-6f);
  r = fmaf(r, s, 2.03457112e-5f);
  r = fmaf(r, s, -2.10521728e-4f);
  r = fmaf(r, s, 2.11580913e-3f);
  r = fmaf(r, s, -2.22220998e-2f);
  r = fmaf(r, s, 8.33333284e-2f);
  r = fmaf(r, x, 0.25f * x);
  o[t] = r;
}
