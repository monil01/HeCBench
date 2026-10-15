#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
#include <cmath>

#include "gels_types.h"

template <typename T>
__device__ T addv(T a, T b) { return a + b; }
template <typename T>
__device__ T subv(T a, T b) { return a - b; }
template <typename T>
__device__ T mulv(T a, T b) { return a * b; }
template <typename T>
__device__ T divv(T a, T b) { return a / b; }
template <typename T>
__device__ double absv(T a) { return fabs(static_cast<double>(a)); }

template <>
__device__ ComplexF addv(ComplexF a, ComplexF b) { return {a.real + b.real, a.imag + b.imag}; }
template <>
__device__ ComplexF subv(ComplexF a, ComplexF b) { return {a.real - b.real, a.imag - b.imag}; }
template <>
__device__ ComplexF mulv(ComplexF a, ComplexF b) { return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real}; }
template <>
__device__ ComplexF divv(ComplexF a, ComplexF b)
{
  float den = b.real * b.real + b.imag * b.imag;
  return {(a.real * b.real + a.imag * b.imag) / den, (a.imag * b.real - a.real * b.imag) / den};
}
template <>
__device__ double absv(ComplexF a) { return sqrt(static_cast<double>(a.real * a.real + a.imag * a.imag)); }

template <>
__device__ ComplexD addv(ComplexD a, ComplexD b) { return {a.real + b.real, a.imag + b.imag}; }
template <>
__device__ ComplexD subv(ComplexD a, ComplexD b) { return {a.real - b.real, a.imag - b.imag}; }
template <>
__device__ ComplexD mulv(ComplexD a, ComplexD b) { return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real}; }
template <>
__device__ ComplexD divv(ComplexD a, ComplexD b)
{
  double den = b.real * b.real + b.imag * b.imag;
  return {(a.real * b.real + a.imag * b.imag) / den, (a.imag * b.real - a.real * b.imag) / den};
}
template <>
__device__ double absv(ComplexD a) { return sqrt(a.real * a.real + a.imag * a.imag); }

template <typename T>
__device__ void solve5(const T* A_in, T* B_out)
{
  T a[5][5];
  T b[5];
  for (int row = 0; row < 5; row++) {
    b[row] = B_out[row];
    for (int col = 0; col < 5; col++) {
      a[row][col] = A_in[col * 5 + row];
    }
  }

  for (int k = 0; k < 5; k++) {
    int pivot = k;
    double best = absv(a[k][k]);
    for (int row = k + 1; row < 5; row++) {
      double v = absv(a[row][k]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (pivot != k) {
      for (int col = 0; col < 5; col++) {
        T tmp = a[k][col];
        a[k][col] = a[pivot][col];
        a[pivot][col] = tmp;
      }
      T tmp = b[k];
      b[k] = b[pivot];
      b[pivot] = tmp;
    }
    T diag = a[k][k];
    for (int row = k + 1; row < 5; row++) {
      T factor = divv(a[row][k], diag);
      for (int col = k; col < 5; col++) {
        a[row][col] = subv(a[row][col], mulv(factor, a[k][col]));
      }
      b[row] = subv(b[row], mulv(factor, b[k]));
    }
  }

  T x[5];
  for (int row = 4; row >= 0; row--) {
    T sum = b[row];
    for (int col = row + 1; col < 5; col++) {
      sum = subv(sum, mulv(a[row][col], x[col]));
    }
    x[row] = divv(sum, a[row][row]);
  }
  for (int row = 0; row < 5; row++) {
    B_out[row] = x[row];
  }
}

extern "C" __global__ void gels_float(const float* A, float* B)
{
  int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < 2) solve5(A + batch * 25, B + batch * 5);
}

extern "C" __global__ void gels_double(const double* A, double* B)
{
  int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < 2) solve5(A + batch * 25, B + batch * 5);
}

extern "C" __global__ void gels_cfloat(const ComplexF* A, ComplexF* B)
{
  int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < 2) solve5(A + batch * 25, B + batch * 5);
}

extern "C" __global__ void gels_cdouble(const ComplexD* A, ComplexD* B)
{
  int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < 2) solve5(A + batch * 25, B + batch * 5);
}
