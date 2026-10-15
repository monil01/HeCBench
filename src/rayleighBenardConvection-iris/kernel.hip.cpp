#include <hip/hip_runtime.h>
#include "parameters.hpp"
extern "C" __global__ void ElemMultOmega(float* A, const float* B, size_t offsetA, size_t offsetB) {
 A+=offsetA;B+=offsetB;

  if(THREADID < (M-2)*(N-2) ) {
    A[THREADID] = A[THREADID]*B[THREADID];
  }
}
extern "C" __global__ void ElemMultT(float* A, const float* B, size_t offsetA, size_t offsetB) {
 A+=offsetA;B+=offsetB;

  if(THREADID < (M-2)*(N)) {
    A[THREADID] = A[THREADID]*B[THREADID];
  }
}
extern "C" __global__ void ElemMultNu(float* A, const float* B, size_t offsetA, size_t offsetB) {
 A+=offsetA;B+=offsetB;

  if(THREADID < N) {
    A[THREADID] = A[THREADID]*B[THREADID];
  }
}
extern "C" __global__ void SubOne(float* A, float unused, size_t offsetA) {
 A+=offsetA;

  if(THREADID < N) {
    A[THREADID] = A[THREADID] - 1.f;
  }
}
extern "C" __global__ void AddOne(float* A, float unused, size_t offsetA) {
 A+=offsetA;

  if(THREADID < N) {
    A[THREADID] = A[THREADID] + 1.f;
  }
}
extern "C" __global__ void AddX(float* A, const float x, size_t offsetA) {
 A+=offsetA;

  if(THREADID < (M-2)*N) {
    A[THREADID] = A[THREADID] + x;
  }
}
extern "C" __global__ void Updatedt(int ptru, const float*__restrict__ u, int ptrv,
                         const float*__restrict__ v, float* dt, size_t offsetU, size_t offsetV, size_t offsetDt) {
 u+=offsetU;v+=offsetV;dt+=offsetDt;

  dt[0] = max(abs(u[ptru - 1]),abs(v[ptrv - 1]));
  dt[0] = min(DX/dt[0],DX2/4.f);
}
