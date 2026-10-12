#include "mv.h"

#include <cassert>
#include <cmath>
#include <cstdlib>

void init_vector(REAL* vector, size_t m)
{
  for (size_t i = 0; i < m; i++) vector[i] = static_cast<REAL>(drand48());
}

void init_matrix(REAL* matrix, size_t num_rows, size_t nnz)
{
  size_t n = num_rows * num_rows;
  REAL* d = static_cast<REAL*>(std::malloc(n * sizeof(REAL)));

  std::srand(123);
  for (size_t i = 0; i < n; i++) d[i] = static_cast<REAL>(i);
  for (size_t i = n; i > 0; i--) {
    size_t a = i - 1;
    size_t b = static_cast<size_t>(std::rand()) % i;
    if (a != b) {
      REAL t = d[a];
      d[a] = d[b];
      d[b] = t;
    }
  }

  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < num_rows; j++) {
      matrix[i * num_rows + j] = (d[i * num_rows + j] >= nnz)
                                     ? static_cast<REAL>(0)
                                     : static_cast<REAL>(drand48() + 1);
    }
  }

  std::free(d);
}

void init_csr(size_t* row_indices, REAL* values, size_t* col_indices, REAL* matrix,
              size_t num_rows, size_t nnz)
{
  row_indices[num_rows] = nnz;
  row_indices[0] = 0;
  size_t* non_zero_elements = static_cast<size_t*>(std::malloc(num_rows * sizeof(size_t)));

  size_t tmp = 0;
  for (size_t i = 0; i < num_rows; i++) {
    size_t nnz_per_row = 0;
    for (size_t j = 0; j < num_rows; j++) {
      if (matrix[i * num_rows + j] != 0) {
        values[tmp] = matrix[i * num_rows + j];
        col_indices[tmp] = j;
        tmp++;
        nnz_per_row++;
      }
    }
    non_zero_elements[i] = nnz_per_row;
  }
  assert(tmp == nnz);

  for (size_t i = 1; i < num_rows; i++) {
    row_indices[i] = row_indices[i - 1] + non_zero_elements[i - 1];
  }

  std::free(non_zero_elements);
}

void mv_csr_serial(size_t num_rows, const size_t* row_indices, const size_t* col_indices,
                   const REAL* values, const REAL* x, REAL* y)
{
  for (size_t row = 0; row < num_rows; row++) {
    size_t row_start = row_indices[row];
    size_t row_end = row_indices[row + 1];
    REAL dot = 0;
    for (size_t i = row_start; i < row_end; i++) dot += values[i] * x[col_indices[i]];
    y[row] = dot;
  }
}

float check(REAL* A, REAL* B, size_t n)
{
  double diffsum = 0;
  double sum = 0;
  for (size_t i = 0; i < n; i++) {
    diffsum += std::fabs(A[i] - B[i]);
    sum += std::fabs(B[i]);
  }
  return static_cast<float>(diffsum / sum);
}
