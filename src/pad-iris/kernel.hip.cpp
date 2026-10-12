#include <hip/hip_runtime.h>

extern "C" __global__ void padding_kernel(double* matrix_out, const double* matrix, int n,
                                           int m, int pad, int matrix_size) {
  int pos = matrix_size - 1 - static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (pos < 0 || pos >= matrix_size) return;

  int out_cols = n + pad;
  int row = pos / out_cols;
  int col = pos % out_cols;
  if (row >= m) return;
  if (col < n) {
    matrix_out[pos] = matrix[row * n + col];
  } else {
    matrix_out[pos] = 0.0;
  }
}
