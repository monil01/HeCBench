#pragma once
#if defined(__HIPCC__)
#include "hip/hip_runtime.h"
#include <hip/hip_complex.h>

extern "C" __global__ void ref_complex_float (char* checkSum, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  uint64_t seed = 1ULL;
  seed = fast_forward_LCG(seed, i);
  float r1 = LCG_random_double(&seed);
  float r2 = LCG_random_double(&seed);
  float r3 = LCG_random_double(&seed);
  float r4 = LCG_random_double(&seed);

  hipFloatComplex z1 = make_hipFloatComplex(r1, r2);
  hipFloatComplex z2 = make_hipFloatComplex(r3, r4);

  char s = fabsf(hipCabsf(Cmulf(z1, z2)) - hipCabsf(z1) * hipCabsf(z2)) < 1e-3f;

  s += fabsf(hipCabsf(hipCaddf(z1, z2)) * hipCabsf(hipCaddf(z1 , z2)) -
             hipCrealf(hipCmulf(hipCaddf(z1, z2) , hipCaddf(hipConjf(z1), hipConjf(z2))))) < 1e-3f;

  s += fabsf(hipCabsf(hipCsubf(z1, z2)) * hipCabsf(hipCsubf(z1 , z2)) -
             hipCrealf(hipCmulf(hipCsubf(z1, z2) , hipCsubf(hipConjf(z1), hipConjf(z2))))) < 1e-3f;

  s += fabsf(hipCrealf(hipCaddf(hipCmulf(z1, hipConjf(z2)) , hipCmulf(z2, hipConjf(z1)))) -
             2.0f * (hipCrealf(z1) * hipCrealf(z2) + hipCimagf(z1) * hipCimagf(z2))) < 1e-3f;

  s += fabsf(hipCabsf(hipCdivf(hipConjf(z1), z2)) -
             hipCabsf(hipCdivf(hipConjf(z1), hipConjf(z2)))) < 1e-3f;

  checkSum[i] = s;
}

extern "C" __global__ void ref_complex_double (char* checkSum, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  uint64_t seed = 1ULL;
  seed = fast_forward_LCG(seed, i);
  double r1 = LCG_random_double(&seed);
  double r2 = LCG_random_double(&seed);
  double r3 = LCG_random_double(&seed);
  double r4 = LCG_random_double(&seed);

  hipDoubleComplex z1 = make_hipDoubleComplex(r1, r2);
  hipDoubleComplex z2 = make_hipDoubleComplex(r3, r4);

  char s = fabs(hipCabs(hipCmul(z1, z2)) - hipCabs(z1) * hipCabs(z2)) < 1e-3;

  s += fabs(hipCabs(hipCadd(z1, z2)) * hipCabs(hipCadd(z1 , z2)) -
            hipCreal(hipCmul(hipCadd(z1, z2) , hipCadd(hipConj(z1), hipConj(z2))))) < 1e-3;

  s += fabs(hipCabs(hipCsub(z1, z2)) * hipCabs(hipCsub(z1 , z2)) -
            hipCreal(hipCmul(hipCsub(z1, z2) , hipCsub(hipConj(z1), hipConj(z2))))) < 1e-3;

  s += fabs(hipCreal(hipCadd(hipCmul(z1, hipConj(z2)) , hipCmul(z2, hipConj(z1)))) -
            2.0 * (hipCreal(z1) * hipCreal(z2) + hipCimag(z1) * hipCimag(z2))) < 1e-3;

  s += fabs(hipCabs(hipCdiv(hipConj(z1), z2)) -
            hipCabs(hipCdiv(hipConj(z1), hipConj(z2)))) < 1e-3;

  checkSum[i] = s;
}
#else
#include <cuComplex.h>

extern "C" __global__ void ref_complex_float (char* checkSum, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  uint64_t seed = 1ULL;
  seed = fast_forward_LCG(seed, i);
  float r1 = LCG_random_double(&seed);
  float r2 = LCG_random_double(&seed);
  float r3 = LCG_random_double(&seed);
  float r4 = LCG_random_double(&seed);

  cuFloatComplex z1 = make_cuFloatComplex(r1, r2);
  cuFloatComplex z2 = make_cuFloatComplex(r3, r4);

  char s = fabsf(cuCabsf(Cmulf(z1, z2)) - cuCabsf(z1) * cuCabsf(z2)) < 1e-3f;

  s += fabsf(cuCabsf(cuCaddf(z1, z2)) * cuCabsf(cuCaddf(z1 , z2)) -
             cuCrealf(cuCmulf(cuCaddf(z1, z2) , cuCaddf(cuConjf(z1), cuConjf(z2))))) < 1e-3f;

  s += fabsf(cuCabsf(cuCsubf(z1, z2)) * cuCabsf(cuCsubf(z1 , z2)) -
             cuCrealf(cuCmulf(cuCsubf(z1, z2) , cuCsubf(cuConjf(z1), cuConjf(z2))))) < 1e-3f;

  s += fabsf(cuCrealf(cuCaddf(cuCmulf(z1, cuConjf(z2)) , cuCmulf(z2, cuConjf(z1)))) -
             2.0f * (cuCrealf(z1) * cuCrealf(z2) + cuCimagf(z1) * cuCimagf(z2))) < 1e-3f;

  s += fabsf(cuCabsf(cuCdivf(cuConjf(z1), z2)) -
             cuCabsf(cuCdivf(cuConjf(z1), cuConjf(z2)))) < 1e-3f;

  checkSum[i] = s;
}

extern "C" __global__ void ref_complex_double (char* checkSum, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  uint64_t seed = 1ULL;
  seed = fast_forward_LCG(seed, i);
  double r1 = LCG_random_double(&seed);
  double r2 = LCG_random_double(&seed);
  double r3 = LCG_random_double(&seed);
  double r4 = LCG_random_double(&seed);

  cuDoubleComplex z1 = make_cuDoubleComplex(r1, r2);
  cuDoubleComplex z2 = make_cuDoubleComplex(r3, r4);

  char s = fabs(cuCabs(cuCmul(z1, z2)) - cuCabs(z1) * cuCabs(z2)) < 1e-3;

  s += fabs(cuCabs(cuCadd(z1, z2)) * cuCabs(cuCadd(z1 , z2)) -
            cuCreal(cuCmul(cuCadd(z1, z2) , cuCadd(cuConj(z1), cuConj(z2))))) < 1e-3;

  s += fabs(cuCabs(cuCsub(z1, z2)) * cuCabs(cuCsub(z1 , z2)) -
            cuCreal(cuCmul(cuCsub(z1, z2) , cuCsub(cuConj(z1), cuConj(z2))))) < 1e-3;

  s += fabs(cuCreal(cuCadd(cuCmul(z1, cuConj(z2)) , cuCmul(z2, cuConj(z1)))) -
            2.0 * (cuCreal(z1) * cuCreal(z2) + cuCimag(z1) * cuCimag(z2))) < 1e-3;

  s += fabs(cuCabs(cuCdiv(cuConj(z1), z2)) -
            cuCabs(cuCdiv(cuConj(z1), cuConj(z2)))) < 1e-3;

  checkSum[i] = s;
}
#endif
