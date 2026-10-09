#include <cuda.h>
#include <math.h>

extern "C" __global__ void haccmk_kernel(const int n1, const int n2,
                                          const float* __restrict__ xx,
                                          const float* __restrict__ yy,
                                          const float* __restrict__ zz,
                                          const float* __restrict__ mass,
                                          float* __restrict__ vx2,
                                          float* __restrict__ vy2,
                                          float* __restrict__ vz2,
                                          const float fsrmax,
                                          const float mp_rsm,
                                          const float fcoeff,
                                          const int block_size)
{
  (void)block_size;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n1) return;

  const float ma0 = 0.269327f;
  const float ma1 = -0.0750978f;
  const float ma2 = 0.0114808f;
  const float ma3 = -0.00109313f;
  const float ma4 = 0.0000605491f;
  const float ma5 = -0.00000147177f;

  float xi = 0.0f;
  float yi = 0.0f;
  float zi = 0.0f;
  const float xxi = xx[i];
  const float yyi = yy[i];
  const float zzi = zz[i];

  for (int j = 0; j < n2; j++) {
    const float dxc = xx[j] - xxi;
    const float dyc = yy[j] - yyi;
    const float dzc = zz[j] - zzi;
    const float r2 = dxc * dxc + dyc * dyc + dzc * dzc;
    const float m = mass[j] * (r2 < fsrmax);
    float f = r2 + mp_rsm;
    f = m * (1.0f / (f * sqrtf(f)) -
             (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))));
    xi += f * dxc;
    yi += f * dyc;
    zi += f * dzc;
  }

  vx2[i] += xi * fcoeff;
  vy2[i] += yi * fcoeff;
  vz2[i] += zi * fcoeff;
}
