#ifndef IRIS_SIMPLE_SPMV_MV_H
#define IRIS_SIMPLE_SPMV_MV_H

#include <cstddef>

#define REAL float

void mv_csr_serial(size_t num_rows, const size_t* row_indices, const size_t* col_indices,
                   const REAL* values, const REAL* x, REAL* y);
float check(REAL* A, REAL* B, size_t n);
void init_vector(REAL* vector, size_t m);
void init_matrix(REAL* matrix, size_t num_rows, size_t nnz);
void init_csr(size_t* row_indices, REAL* values, size_t* col_indices, REAL* matrix,
              size_t num_rows, size_t nnz);

#endif
