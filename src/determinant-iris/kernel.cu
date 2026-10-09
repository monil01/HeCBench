#include <cuda_runtime.h>

extern "C" __global__ void determinant_cholesky(const float* a, float* det,
                                                int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  float l[121];
  for (int i = 0; i < n * n; i++) l[i] = 0.0f;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) sum += l[i * n + k] * l[j * n + k];
      if (i == j) {
        l[i * n + j] = sqrtf(a[i * n + i] - sum);
      } else {
        l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
      }
    }
  }

  float diag_product = 1.0f;
  for (int i = 0; i < n; i++) diag_product *= l[i * n + i];
  *det = diag_product * diag_product;
}
