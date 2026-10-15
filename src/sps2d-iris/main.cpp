#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "utils.h"

#define BLOCK_SIZE 256

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("The function converts a sparse matrix into a MxN dense matrix\n");
    std::printf("The sparse matrix is represented in CSR (Compressed Sparse Row) storage format\n");
    std::printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    std::printf("nnz is the number of non-zero elements\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int64_t num_rows = std::atol(argv[1]);
  const int64_t num_cols = std::atol(argv[2]);
  const int64_t h_nnz = std::atol(argv[3]);
  const int repeat = std::atoi(argv[4]);

  const int64_t dense_size = num_rows * num_cols;
  const size_t dense_size_bytes = static_cast<size_t>(dense_size) * sizeof(float);
  const size_t value_size_bytes = static_cast<size_t>(h_nnz) * sizeof(float);
  const size_t colidx_size_bytes = static_cast<size_t>(h_nnz) * sizeof(int64_t);
  const size_t rowidx_size_bytes = static_cast<size_t>(num_rows + 1) * sizeof(int64_t);

  float* h_dense = static_cast<float*>(std::malloc(dense_size_bytes));
  float* h_dense_result = static_cast<float*>(std::calloc(static_cast<size_t>(dense_size), sizeof(float)));
  float* h_csr_values = static_cast<float*>(std::malloc(value_size_bytes));
  int64_t* h_csr_columns = static_cast<int64_t*>(std::malloc(colidx_size_bytes));
  int64_t* h_csr_offsets = static_cast<int64_t*>(std::malloc(rowidx_size_bytes));
  if (!h_dense || !h_dense_result || !h_csr_values || !h_csr_columns || !h_csr_offsets) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_dense);
    std::free(h_dense_result);
    std::free(h_csr_values);
    std::free(h_csr_columns);
    std::free(h_csr_offsets);
    platform.finalize();
    return 1;
  }

  std::printf("Initializing host matrices..\n");
  init_matrix(h_dense, num_rows, num_cols, h_nnz);
  init_csr(h_csr_offsets, h_csr_values, h_csr_columns, h_dense, num_rows, num_cols, h_nnz);

  iris::DMem mem_dense(h_dense_result, dense_size_bytes);
  iris::DMem mem_csr_offsets(h_csr_offsets, rowidx_size_bytes);
  iris::DMem mem_csr_columns(h_csr_columns, colidx_size_bytes);
  iris::DMem mem_csr_values(h_csr_values, value_size_bytes);

  const size_t gws[1] = {
    static_cast<size_t>(((num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)
  };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[6] = {
      &mem_dense,
      &mem_csr_offsets,
      &mem_csr_columns,
      &mem_csr_values,
      const_cast<int64_t*>(&num_rows),
      const_cast<int64_t*>(&num_cols)
    };
    int pinfo[6] = { iris_rw, iris_r, iris_r, iris_r, sizeof(num_rows), sizeof(num_cols) };
    task.kernel("sparse_to_dense_csr", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 6, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_dense);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of SparseToDense_convert : %f (us)\n", (time * 1e-3f) / repeat);

  int correct = 1;
  int64_t nnz = 0;
  for (int64_t i = 0; i < dense_size; i++) {
    if (h_dense_result[i] != 0.0f) {
      nnz++;
    }

    if (h_dense[i] != h_dense_result[i]) {
      std::printf("@%ld: %f != %f\n", i, h_dense[i], h_dense_result[i]);
      correct = 0;
      break;
    }
  }

  if (nnz != h_nnz) {
    std::printf("nnz: %ld != %ld\n", h_nnz, nnz);
    correct = 0;
  }

  if (correct) {
    std::printf("sparse2dense_csr_example test PASSED\n");
  } else {
    std::printf("sparse2dense_csr_example test FAILED: wrong result\n");
  }

  std::free(h_dense);
  std::free(h_dense_result);
  std::free(h_csr_values);
  std::free(h_csr_columns);
  std::free(h_csr_offsets);
  platform.finalize();
  return correct ? 0 : 1;
}
