extern "C" __global__ void csr2csc_serial(
    const int* __restrict__ A_offsets,
    const int* __restrict__ A_columns,
    const float* __restrict__ A_values,
    int* __restrict__ B_offsets,
    int* __restrict__ B_rows,
    float* __restrict__ B_values,
    int* __restrict__ next,
    int A_num_rows,
    int A_num_cols,
    int A_nnz)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  for (int i = 0; i <= A_num_cols; i++) {
    B_offsets[i] = 0;
    next[i] = 0;
  }

  for (int idx = 0; idx < A_nnz; idx++) {
    int col = A_columns[idx];
    B_offsets[col + 1]++;
  }

  for (int i = 0; i < A_num_cols; i++) {
    B_offsets[i + 1] += B_offsets[i];
    next[i] = B_offsets[i];
  }

  for (int row = 0; row < A_num_rows; row++) {
    for (int idx = A_offsets[row]; idx < A_offsets[row + 1]; idx++) {
      int col = A_columns[idx];
      int pos = next[col]++;
      B_rows[pos] = row;
      B_values[pos] = A_values[idx];
    }
  }
}
