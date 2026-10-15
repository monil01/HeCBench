#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void sparse_to_dense_csr(
    float* dense,
    const int64_t* csr_offsets,
    const int64_t* csr_columns,
    const float* csr_values,
    int64_t num_rows,
    int64_t num_cols)
{
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_rows) {
    return;
  }

  const int64_t begin = csr_offsets[row];
  const int64_t end = csr_offsets[row + 1];
  for (int64_t idx = begin; idx < end; idx++) {
    dense[row * num_cols + csr_columns[idx]] = csr_values[idx];
  }
}
