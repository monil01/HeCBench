#include <iris/iris.hpp>

#include "mv.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static long run_dense(int repeat, int bs, size_t num_rows, const REAL* x, REAL* matrix, REAL* y)
{
  iris::DMem mem_x(const_cast<REAL*>(x), num_rows * sizeof(REAL));
  iris::DMem mem_matrix(matrix, num_rows * num_rows * sizeof(REAL));
  iris::DMem mem_y(y, num_rows * sizeof(REAL));
  size_t gws[1] = {((num_rows + static_cast<size_t>(bs) - 1) / static_cast<size_t>(bs)) *
                   static_cast<size_t>(bs)};
  size_t lws[1] = {static_cast<size_t>(bs)};
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {&num_rows, &mem_matrix, &mem_x, &mem_y};
    int pinfo[4] = {sizeof(num_rows), iris_r, iris_r, iris_w};
    task.kernel("mv_dense", 1, nullptr, gws, lws, 4, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_y);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static long run_csr(int repeat, int bs, size_t num_rows, const size_t* row_indices,
                    const size_t* col_indices, const REAL* values, const REAL* x, size_t nnz,
                    REAL* y)
{
  iris::DMem mem_row(const_cast<size_t*>(row_indices), (num_rows + 1) * sizeof(size_t));
  iris::DMem mem_col(const_cast<size_t*>(col_indices), nnz * sizeof(size_t));
  iris::DMem mem_values(const_cast<REAL*>(values), nnz * sizeof(REAL));
  iris::DMem mem_x(const_cast<REAL*>(x), num_rows * sizeof(REAL));
  iris::DMem mem_y(y, num_rows * sizeof(REAL));
  size_t gws[1] = {((num_rows + static_cast<size_t>(bs) - 1) / static_cast<size_t>(bs)) *
                   static_cast<size_t>(bs)};
  size_t lws[1] = {static_cast<size_t>(bs)};
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[6] = {&num_rows, &mem_row, &mem_col, &mem_values, &mem_x, &mem_y};
    int pinfo[6] = {sizeof(num_rows), iris_r, iris_r, iris_r, iris_r, iris_w};
    task.kernel("mv_csr", 1, nullptr, gws, lws, 6, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_y);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static size_t prev_power_of_2(size_t v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v >> 1;
}

static long run_vector_csr(int repeat, int bs, size_t num_rows, const size_t* row_indices,
                           const size_t* col_indices, const REAL* values, const REAL* x,
                           size_t nnz, REAL* y)
{
  iris::DMem mem_row(const_cast<size_t*>(row_indices), (num_rows + 1) * sizeof(size_t));
  iris::DMem mem_col(const_cast<size_t*>(col_indices), nnz * sizeof(size_t));
  iris::DMem mem_values(const_cast<REAL*>(values), nnz * sizeof(REAL));
  iris::DMem mem_x(const_cast<REAL*>(x), num_rows * sizeof(REAL));
  iris::DMem mem_y(y, num_rows * sizeof(REAL));

  int nnz_per_row = static_cast<int>(nnz / num_rows);
  int threads_per_row = static_cast<int>(prev_power_of_2(static_cast<size_t>(nnz_per_row)));
  threads_per_row = threads_per_row > 32 ? 32 : threads_per_row;
  if (threads_per_row < 2) threads_per_row = 2;
  int rows_per_block = bs / threads_per_row;
  if (rows_per_block == 0) rows_per_block = 1;
  int num_blocks = static_cast<int>((num_rows + static_cast<size_t>(rows_per_block) - 1) /
                                    static_cast<size_t>(rows_per_block));
  size_t gws[2] = {static_cast<size_t>(num_blocks * threads_per_row),
                   static_cast<size_t>(rows_per_block)};
  size_t lws[2] = {static_cast<size_t>(threads_per_row), static_cast<size_t>(rows_per_block)};
  const char* kernel_name = threads_per_row <= 2    ? "vector_mv_csr_2"
                            : threads_per_row <= 4  ? "vector_mv_csr_4"
                            : threads_per_row <= 8  ? "vector_mv_csr_8"
                            : threads_per_row <= 16 ? "vector_mv_csr_16"
                                                    : "vector_mv_csr_32";

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[6] = {&num_rows, &mem_row, &mem_col, &mem_values, &mem_x, &mem_y};
    int pinfo[6] = {sizeof(num_rows), iris_r, iris_r, iris_r, iris_r, iris_w};
    task.kernel(kernel_name, 2, nullptr, gws, lws, 6, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_y);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage %s <number of non-zero elements> <number of rows in a square matrix> <repeat>\n",
                argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  size_t nnz = static_cast<size_t>(std::atol(argv[1]));
  size_t num_rows = static_cast<size_t>(std::atol(argv[2]));
  int repeat = std::atoi(argv[3]);
  size_t num_elems = num_rows * num_rows;
  assert(nnz > 0);
  assert(num_rows > 0);
  assert(nnz <= num_elems);

  size_t* row_indices = static_cast<size_t*>(std::malloc((num_rows + 1) * sizeof(size_t)));
  size_t* col_indices = static_cast<size_t*>(std::malloc(nnz * sizeof(size_t)));
  REAL* values = static_cast<REAL*>(std::malloc(nnz * sizeof(REAL)));
  REAL* x = static_cast<REAL*>(std::malloc(num_rows * sizeof(REAL)));
  REAL* y[4];
  for (int i = 0; i < 4; i++) y[i] = static_cast<REAL*>(std::malloc(num_rows * sizeof(REAL)));
  REAL* matrix = static_cast<REAL*>(std::malloc(num_elems * sizeof(REAL)));

  srand48(1 << 12);
  init_matrix(matrix, num_rows, nnz);
  init_vector(x, num_rows);
  init_csr(row_indices, values, col_indices, matrix, num_rows, nnz);
  mv_csr_serial(num_rows, row_indices, col_indices, values, x, y[0]);

  std::printf("Number of non-zero elements: %lu\n", nnz);
  std::printf("Number of rows in a square matrix: %lu\n", num_rows);
  std::printf("Sparsity: %lf%%\n", (num_elems - nnz) * 1.0 / num_elems * 100.0);

  long elapsed[3];
  for (int bs = 32; bs <= 1024; bs *= 2) {
    std::printf("\nThread block size: %d\n", bs);
    elapsed[0] = run_dense(repeat, bs, num_rows, x, matrix, y[1]);
    elapsed[1] = run_csr(repeat, bs, num_rows, row_indices, col_indices, values, x, nnz, y[2]);
    elapsed[2] = run_vector_csr(repeat, bs, num_rows, row_indices, col_indices, values, x, nnz, y[3]);

    std::printf("Average dense, sparse, and vector sparse kernel execution time (ms):");
    for (int i = 0; i < 3; i++) std::printf(" %lf", elapsed[i] * 1e-6 / repeat);
    std::printf("\nError rate:");
    for (int i = 1; i < 4; i++) std::printf(" %f", check(y[0], y[i], num_rows));
    std::printf("\n");
  }

  std::free(row_indices);
  std::free(col_indices);
  std::free(values);
  std::free(x);
  std::free(matrix);
  for (int i = 0; i < 4; i++) std::free(y[i]);
  platform.finalize();
  return 0;
}
