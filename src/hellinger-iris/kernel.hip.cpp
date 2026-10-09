#include <hip/hip_runtime.h>

#ifdef DOUBLE_PRECISION
  #define SQRT sqrt
  #define FP double
#else
  #define SQRT sqrtf
  #define FP float
#endif

extern "C" __global__ void hellinger(const FP* __restrict__ a,
                                     const FP* __restrict__ b,
                                     FP* __restrict__ c, const int m,
                                     const int n, const int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < k && row < m) {
    FP sum = 0;
    for (int i = 0; i < n; i++) {
      sum += SQRT(a[row * n + i] * b[i * k + col]);
    }
    const FP value = (FP)1.0 - sum;
    const FP gate = (!signbit(value));
    c[row * k + col] = SQRT(gate * value);
  }
}
