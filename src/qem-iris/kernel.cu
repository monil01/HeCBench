#include <cuda_runtime.h>
#include <cmath>
extern "C" __global__ void QRdel(int n, const float *A, const float *B, const float *C,
                      const float *D, float *__restrict__ b,
                      float *__restrict__ c, float *__restrict__ d,
                      float *__restrict__ Q, float *__restrict__ R,
                      float *__restrict__ Qint, float *__restrict__ Rint,
                      float *__restrict__ del) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {

    b[i] = 0.75f * (B[i] / A[i]);
    c[i] = 0.50f * (C[i] / A[i]);
    d[i] = 0.25f * (D[i] / A[i]);

    Q[i] = (c[i] / 3.f) - ((b[i] * b[i]) / 9.f);
    R[i] = (b[i] * c[i]) / 6.f - (b[i] * b[i] * b[i]) / 27.f - 0.5f * d[i];

    // round Q and R to get around problems caused by floating point precision
    Q[i] = roundf(Q[i] * 1E5f) / 1E5f;
    R[i] = roundf(R[i] * 1E5f) / 1E5f;

    Qint[i] = (Q[i] * Q[i] * Q[i]);
    Rint[i] = (R[i] * R[i]);

    del[i] = Rint[i] + Qint[i];
    // del[i] = (R[i] * R[i]) + (Q[i] * Q[i] * Q[i]); // why not just Q*Q*Q +
    // R*R? Heisenbug. Heisenbug in release code
  }
}

extern "C" __global__ void QuarticSolver(int n, const float *A, const float *B,
                              const float *C, const float *D, const float *b,
                              const float *Q, const float *R, const float *del,
                              float *__restrict__ theta,
                              float *__restrict__ sqrtQ, float *__restrict__ x1,
                              float *__restrict__ x2, float *__restrict__ x3,
                              float *__restrict__ temp,
                              float *__restrict__ min) {
  // solver for finding minimum (xmin) for f(x) = Ax^4 + Bx^3 + Cx^2 + Dx + E
  // undefined behaviour if A=0

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // comparing against 1E-5 to deal with potential problems with comparing
    // floats to zero
    if (del[i] <= 1E-5f) { // all 3 roots real

      /*sqrtQ = 2 * sqrt(-Q);
         theta = acos(R / (sqrtQ ^ 3));

         x1 = 2 * (sqrtQ*cos(theta / 3) - b/3;
         x2 = 2 * (sqrtQ*cos((theta + 2 * pi) / 3) - b/3);
         x3 = 2 * (sqrtQ*cos((theta + 4 * pi) / 3) - b/3);*/

      theta[i] = acosf((R[i] / sqrtf(-(Q[i] * Q[i] * Q[i]))));
      sqrtQ[i] = 2.f * sqrtf(-Q[i]);

      x1[i] = ((sqrtQ[i] * cosf((theta[i]) / 3.f)) - (b[i] / 3.f));
      x2[i] = ((sqrtQ[i] * cosf((theta[i] + 2.f * 3.1415927f) / 3.f)) -
               (b[i] / 3.f));
      x3[i] = ((sqrtQ[i] * cosf((theta[i] + 4.f * 3.1415927f) / 3.f)) -
               (b[i] / 3.f));

      // unrolled bubble sort
      if (x1[i] < x2[i]) {
        temp[i] = x1[i];
        x1[i] = x2[i];
        x2[i] = temp[i];
      } // { swap(x1[i], x2[i]); }//swap
      if (x2[i] < x3[i]) {
        temp[i] = x2[i];
        x2[i] = x3[i];
        x3[i] = temp[i];
      } //{ swap(x2[i], x3[i]); }//swap
      if (x1[i] < x2[i]) {
        temp[i] = x1[i];
        x1[i] = x2[i];
        x2[i] = temp[i];
      } //{ swap(x1[i], x2[i]); }//swap

      min[i] =
          A[i] *
                          ((x1[i] * x1[i] * x1[i] * x1[i]) -
                           (x3[i] * x3[i] * x3[i] * x3[i])) /
                          4.f +
                      B[i] *
                          ((x1[i] * x1[i] * x1[i]) - (x3[i] * x3[i] * x3[i])) /
                          3.f +
                      C[i] * ((x1[i] * x1[i]) - (x3[i] * x3[i])) / 2.f +
                      D[i] * (x1[i] - x3[i]) <=
                  0.f
              ? x1[i]
              : x3[i];

    }

    // if (del[i] > 0) { // only 1 real root
    else {

      /*S = (R + sqrtD)^(1 / 3);
         T = (R - sqrtD)^(1 / 3);
         x = S + T - b/3;*/

      x1[i] = cbrtf((R[i] + sqrtf(del[i]))) + cbrtf((R[i] - sqrtf(del[i]))) -
              (b[i] / 3.f); // real root

      // complex conjugate roots not relevant for minimisation

      x2[i] = 0;
      x3[i] = 0;

      min[i] = x1[i];
    }
  }
}
