#include <hip/hip_runtime.h>

extern "C" __global__ void spnnz_count_rows(int* row_counts,
                                             const float* dense, long num_rows,
                                             long num_cols) {
  const long row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  int count = 0;
  const long offset = row * num_cols;
  for (long col = 0; col < num_cols; col++) {
    if (dense[offset + col] != 0.0f) count++;
  }
  row_counts[row] = count;
}
