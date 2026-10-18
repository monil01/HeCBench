// Study repair: sole-writer guard only; original template is preserved in source.
template <int BS>
__global__ void vector_mv_csr_defined(const size_t num_rows,
                              const size_t *row_indices,
                              const size_t *col_indices,
                              const REAL *values,
                              const REAL *x,
                                    REAL *y)
{
  size_t m = blockIdx.x * blockDim.y + threadIdx.y;
  if (m < num_rows) {
    size_t row_start = row_indices[m];
    size_t row_end = row_indices[m+1];

    REAL temp = 0;
    for(size_t n = row_start + threadIdx.x; n < row_end; n += BS){
      temp += values[n] * x[col_indices[n]];
    }
    #pragma unroll
    for (int i = BS >> 1; i > 0; i >>= 1)
      temp += __shfl_down_sync(0xFFFFFFFF, temp, i, BS);

    if (threadIdx.x == 0) y[m] = temp;
  }
}


