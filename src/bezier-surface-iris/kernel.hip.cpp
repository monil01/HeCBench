#include <hip/hip_runtime.h>
#include <math.h>

#if DOUBLE_PRECISION
#define FLOAT double
#else
#define FLOAT float
#endif

typedef struct {
  FLOAT x;
  FLOAT y;
  FLOAT z;
} XYZ;

__host__ __device__
inline FLOAT BezierBlend(int k, FLOAT mu, int n)
{
  int nn = n;
  int kn = k;
  int nkn = n - k;
  FLOAT blend = 1;
  while (nn >= 1) {
    blend *= nn;
    nn--;
    if (kn > 1) {
      blend /= (FLOAT)kn;
      kn--;
    }
    if (nkn > 1) {
      blend /= (FLOAT)nkn;
      nkn--;
    }
  }
  if (k > 0) {
    blend *= pow(mu, (FLOAT)k);
  }
  if (n - k > 0) {
    blend *= pow(1 - mu, (FLOAT)(n - k));
  }
  return blend;
}

extern "C" __global__
void BezierGPU(const XYZ *inp, XYZ *outp, const int NI, const int NJ,
               const int RESOLUTIONI, const int RESOLUTIONJ)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= RESOLUTIONI) {
    return;
  }

  FLOAT mui = i / (FLOAT)(RESOLUTIONI - 1);
  for (int j = 0; j < RESOLUTIONJ; j++) {
    FLOAT muj = j / (FLOAT)(RESOLUTIONJ - 1);
    XYZ out = {0, 0, 0};
    for (int ki = 0; ki <= NI; ki++) {
      FLOAT bi = BezierBlend(ki, mui, NI);
      for (int kj = 0; kj <= NJ; kj++) {
        FLOAT bj = BezierBlend(kj, muj, NJ);
        out.x += inp[ki * (NJ + 1) + kj].x * bi * bj;
        out.y += inp[ki * (NJ + 1) + kj].y * bi * bj;
        out.z += inp[ki * (NJ + 1) + kj].z * bi * bj;
      }
    }
    outp[i * RESOLUTIONJ + j] = out;
  }
}
