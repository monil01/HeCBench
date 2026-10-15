#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <cstring>

static void init_matrix(float* matrix, int num_rows, int num_cols, int nnz)
{
  int n = num_rows * num_cols;
  float* d = static_cast<float*>(std::malloc(n * sizeof(float)));

  std::srand(123);
  for (int i = 0; i < n; i++) d[i] = static_cast<float>(i);
  for (int i = n; i > 0; i--) {
    int a = i - 1;
    int b = std::rand() % i;
    if (a != b) {
      auto t = d[a];
      d[a] = d[b];
      d[b] = t;
    }
  }

  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      matrix[i * num_cols + j] = (d[i * num_cols + j] >= nnz) ? 0 : static_cast<float>(drand48() + 1);
    }
  }

  std::free(d);
}

static void init_csr(int* row_indices, float* values, int* col_indices, float* matrix, int num_rows, int num_cols, int nnz)
{
  row_indices[num_rows] = nnz;
  row_indices[0] = 0;
  int* non_zero_elements = static_cast<int*>(std::malloc(num_rows * sizeof(int)));

  int tmp = 0;
  for (int i = 0; i < num_rows; i++) {
    int nnz_per_row = 0;
    for (int j = 0; j < num_cols; j++) {
      if (matrix[i * num_cols + j] != 0) {
        values[tmp] = matrix[i * num_cols + j];
        col_indices[tmp] = j;
        tmp++;
        nnz_per_row++;
      }
    }
    non_zero_elements[i] = nnz_per_row;
  }

  for (int i = 1; i < num_rows; i++) {
    row_indices[i] = row_indices[i - 1] + non_zero_elements[i - 1];
  }

  std::free(non_zero_elements);
}

static void transpose_dense(const float* A, float* B, int A_num_cols, int A_num_rows, int transpose)
{
  if (!transpose) {
    std::memcpy(B, A, A_num_cols * A_num_rows * sizeof(float));
  } else {
    for (int i = 0; i < A_num_cols; i++) {
      for (int j = 0; j < A_num_rows; j++) {
        B[i * A_num_rows + j] = A[j * A_num_cols + i];
      }
    }
  }
}

#endif
