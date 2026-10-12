#include <hip/hip_runtime.h>

extern "C" __global__ void amgmk_relax(double* u, const double* a,
                                        const int* row, const int* col,
                                        const double* f, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  if (a[row[i]] != 0.0) {
    double residual = f[i];
    for (int p = row[i] + 1; p < row[i + 1]; ++p) residual -= a[p] * u[col[p]];
    u[i] = residual / a[row[i]];
  }
}

