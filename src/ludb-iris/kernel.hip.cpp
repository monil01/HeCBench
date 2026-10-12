#include <hip/hip_runtime.h>

#include <cmath>

#define N 48

extern "C" __global__ void ludb_getrf_batched(float* A, int* pivot, int* info, int batch_size)
{
  int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= batch_size) return;

  float* mat = A + static_cast<size_t>(batch) * N * N;
  int* piv = pivot + batch * N;
  int local_info = 0;

  for (int k = 0; k < N; k++) {
    int p = k;
    float max_abs = fabsf(mat[k * N + k]);
    for (int row = k + 1; row < N; row++) {
      float v = fabsf(mat[k * N + row]);
      if (v > max_abs) {
        max_abs = v;
        p = row;
      }
    }

    piv[k] = p + 1;
    if (max_abs == 0.0f && local_info == 0) {
      local_info = k + 1;
      continue;
    }

    if (p != k) {
      for (int col = 0; col < N; col++) {
        float tmp = mat[col * N + k];
        mat[col * N + k] = mat[col * N + p];
        mat[col * N + p] = tmp;
      }
    }

    float diag = mat[k * N + k];
    for (int row = k + 1; row < N; row++) {
      mat[k * N + row] /= diag;
    }

    for (int col = k + 1; col < N; col++) {
      float ukj = mat[col * N + k];
      for (int row = k + 1; row < N; row++) {
        mat[col * N + row] -= mat[k * N + row] * ukj;
      }
    }
  }

  info[batch] = local_info;
}
