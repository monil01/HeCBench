#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <cstdlib>

void init_matrix(float* matrix, int64_t num_rows, int64_t num_cols, int64_t nnz)
{
  int64_t n = num_rows * num_cols;

  float* d = static_cast<float*>(std::malloc(n * sizeof(float)));

  std::srand(123);
  for (int64_t i = 0; i < n; i++) {
    d[i] = static_cast<float>(i);
  }
  for (int64_t i = n; i > 0; i--) {
    int64_t a = i - 1;
    int64_t b = std::rand() % i;
    if (a != b) {
      auto t = d[a];
      d[a] = d[b];
      d[b] = t;
    }
  }

  srand48(123);
  for (int64_t i = 0; i < num_rows; i++) {
    for (int64_t j = 0; j < num_cols; j++) {
      matrix[i * num_cols + j] = (d[i * num_cols + j] >= nnz) ? 0.0f : static_cast<float>(drand48() + 1);
    }
  }

  std::free(d);
}

void init_csr(int64_t* row_indices, float* values,
              int64_t* col_indices, float* matrix,
              int64_t num_rows, int64_t num_cols, int64_t nnz)
{
  row_indices[num_rows] = nnz;
  row_indices[0] = 0;
  int64_t* non_zero_elements = static_cast<int64_t*>(std::malloc(num_rows * sizeof(int64_t)));

  int64_t tmp = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    int64_t nnz_per_row = 0;
    for (int64_t j = 0; j < num_cols; j++) {
      if (matrix[i * num_cols + j] != 0) {
        values[tmp] = matrix[i * num_cols + j];
        col_indices[tmp] = j;
        tmp++;
        nnz_per_row++;
      }
    }
    non_zero_elements[i] = nnz_per_row;
  }

  for (int64_t i = 1; i < num_rows; i++) {
    row_indices[i] = row_indices[i - 1] + non_zero_elements[i - 1];
  }

  std::free(non_zero_elements);
}

#endif
