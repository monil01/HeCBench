#include <hip/hip_runtime.h>
#include <stdint.h>

#include "complex.h"
__device__ double LCG_random_double(uint64_t* seed);
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
#include "reference.h"

__device__ double LCG_random_double(uint64_t* seed)
{
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return static_cast<double>(*seed) / static_cast<double>(m);
}

__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
  const uint64_t m = 9223372036854775808ULL;
  uint64_t a = 2806196910506780709ULL;
  uint64_t c = 1ULL;
  n = n % m;
  uint64_t a_new = 1;
  uint64_t c_new = 0;
  while (n > 0) {
    if (n & 1) {
      a_new *= a;
      c_new = c_new * a + c;
    }
    c *= (a + 1);
    a *= a;
    n >>= 1;
  }
  return (a_new * seed + c_new) % m;
}

extern "C" __global__ void complex_float(char* checkSum, int n)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  uint64_t seed = fast_forward_LCG(1ULL, i);
  const float r1 = LCG_random_double(&seed);
  const float r2 = LCG_random_double(&seed);
  const float r3 = LCG_random_double(&seed);
  const float r4 = LCG_random_double(&seed);
  FloatComplex z1 = make_FloatComplex(r1, r2);
  FloatComplex z2 = make_FloatComplex(r3, r4);
  char s = fabsf(Cabsf(Cmulf(z1, z2)) - Cabsf(z1) * Cabsf(z2)) < 1e-3f;
  s += fabsf(Cabsf(Caddf(z1, z2)) * Cabsf(Caddf(z1, z2)) -
             Crealf(Cmulf(Caddf(z1, z2), Caddf(Conjf(z1), Conjf(z2))))) < 1e-3f;
  s += fabsf(Cabsf(Csubf(z1, z2)) * Cabsf(Csubf(z1, z2)) -
             Crealf(Cmulf(Csubf(z1, z2), Csubf(Conjf(z1), Conjf(z2))))) < 1e-3f;
  s += fabsf(Crealf(Caddf(Cmulf(z1, Conjf(z2)), Cmulf(z2, Conjf(z1)))) -
             2.0f * (Crealf(z1) * Crealf(z2) + Cimagf(z1) * Cimagf(z2))) < 1e-3f;
  s += fabsf(Cabsf(Cdivf(Conjf(z1), z2)) -
             Cabsf(Cdivf(Conjf(z1), Conjf(z2)))) < 1e-3f;
  checkSum[i] = s;
}

extern "C" __global__ void complex_double(char* checkSum, int n)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  uint64_t seed = fast_forward_LCG(1ULL, i);
  const double r1 = LCG_random_double(&seed);
  const double r2 = LCG_random_double(&seed);
  const double r3 = LCG_random_double(&seed);
  const double r4 = LCG_random_double(&seed);
  DoubleComplex z1 = make_DoubleComplex(r1, r2);
  DoubleComplex z2 = make_DoubleComplex(r3, r4);
  char s = fabs(Cabs(Cmul(z1, z2)) - Cabs(z1) * Cabs(z2)) < 1e-3;
  s += fabs(Cabs(Cadd(z1, z2)) * Cabs(Cadd(z1, z2)) -
            Creal(Cmul(Cadd(z1, z2), Cadd(Conj(z1), Conj(z2))))) < 1e-3;
  s += fabs(Cabs(Csub(z1, z2)) * Cabs(Csub(z1, z2)) -
            Creal(Cmul(Csub(z1, z2), Csub(Conj(z1), Conj(z2))))) < 1e-3;
  s += fabs(Creal(Cadd(Cmul(z1, Conj(z2)), Cmul(z2, Conj(z1)))) -
            2.0 * (Creal(z1) * Creal(z2) + Cimag(z1) * Cimag(z2))) < 1e-3;
  s += fabs(Cabs(Cdiv(Conj(z1), z2)) -
            Cabs(Cdiv(Conj(z1), Conj(z2)))) < 1e-3;
  checkSum[i] = s;
}
