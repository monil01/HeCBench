#include <hip/hip_runtime.h>

#define REAL float

extern "C" __global__ void mv_csr(const size_t num_rows,
                                   const size_t* __restrict__ row_indices,
                                   const size_t* __restrict__ col_indices,
                                   const REAL* __restrict__ values,
                                   const REAL* __restrict__ x,
                                   REAL* __restrict__ y)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_rows) {
    size_t row_start = row_indices[i];
    size_t row_end = row_indices[i + 1];
    REAL temp = 0;
    for (size_t n = row_start; n < row_end; n++) temp += values[n] * x[col_indices[n]];
    y[i] = temp;
  }
}

template <int BS>
__device__ void vector_mv_csr_body(const size_t num_rows,
                                   const size_t* __restrict__ row_indices,
                                   const size_t* __restrict__ col_indices,
                                   const REAL* __restrict__ values,
                                   const REAL* __restrict__ x,
                                   REAL* __restrict__ y)
{
  size_t m = blockIdx.x * blockDim.y + threadIdx.y;
  if (m < num_rows) {
    size_t row_start = row_indices[m];
    size_t row_end = row_indices[m + 1];
    REAL temp = 0;
    for (size_t n = row_start + threadIdx.x; n < row_end; n += BS) {
      temp += values[n] * x[col_indices[n]];
    }
#pragma unroll
    for (int i = BS >> 1; i > 0; i >>= 1) {
      temp += __shfl_down(temp, i, BS);
    }
    if (threadIdx.x == 0) y[m] = temp;
  }
}

#define MAKE_VECTOR_CSR(BS)                                                        \
  extern "C" __global__ void vector_mv_csr_##BS(                                  \
      const size_t num_rows, const size_t* __restrict__ row_indices,               \
      const size_t* __restrict__ col_indices, const REAL* __restrict__ values,     \
      const REAL* __restrict__ x, REAL* __restrict__ y)                            \
  {                                                                                \
    vector_mv_csr_body<BS>(num_rows, row_indices, col_indices, values, x, y);      \
  }

MAKE_VECTOR_CSR(2)
MAKE_VECTOR_CSR(4)
MAKE_VECTOR_CSR(8)
MAKE_VECTOR_CSR(16)
MAKE_VECTOR_CSR(32)

extern "C" __global__ void mv_dense(const size_t num_rows,
                                     const REAL* __restrict__ matrix,
                                     const REAL* __restrict__ x,
                                     REAL* __restrict__ y)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_rows) {
    REAL temp = 0;
    for (size_t j = 0; j < num_rows; j++) {
      if (matrix[i * num_rows + j] != static_cast<REAL>(0)) {
        temp += matrix[i * num_rows + j] * x[j];
      }
    }
    y[i] = temp;
  }
}
