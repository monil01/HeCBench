#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "utils.h"

static void launch_csr2csc(
    iris::DMem& mem_A_offsets,
    iris::DMem& mem_A_columns,
    iris::DMem& mem_A_values,
    iris::DMem& mem_B_offsets,
    iris::DMem& mem_B_rows,
    iris::DMem& mem_B_values,
    iris::DMem& mem_next,
    int A_num_rows,
    int A_num_cols,
    int A_nnz)
{
  size_t gws[1] = { 1 };
  size_t lws[1] = { 1 };
  iris::Task task;
  void* params[10] = {
    &mem_A_offsets,
    &mem_A_columns,
    &mem_A_values,
    &mem_B_offsets,
    &mem_B_rows,
    &mem_B_values,
    &mem_next,
    &A_num_rows,
    &A_num_cols,
    &A_nnz
  };
  int pinfo[10] = {
    iris_r,
    iris_r,
    iris_r,
    iris_w,
    iris_w,
    iris_w,
    iris_w,
    sizeof(A_num_rows),
    sizeof(A_num_cols),
    sizeof(A_nnz)
  };
  task.kernel("csr2csc_serial", 1, nullptr, gws, lws, 10, params, pinfo);
  task.flush_out(mem_B_offsets);
  task.flush_out(mem_B_rows);
  task.flush_out(mem_B_values);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    std::printf("The function performs an out-of-place transpose of a sparse matrix into a new one,");
    std::printf("where the input sparse matrix is represented in CSR (Compressed Sparse Row) storage format\n");
    std::printf("Usage %s <M> <K> <nnz> <repeat> <verify>\n", argv[0]);
    std::printf("Input matrix A: M * K\n");
    std::printf("Output matrix B: K * M\n");
    return 1;
  }

  int m = std::atoi(argv[1]);
  int k = std::atoi(argv[2]);
  int a_nnz = std::atoi(argv[3]);
  int repeat = std::atoi(argv[4]);
  int verify = std::atoi(argv[5]);
  if (m <= 0 || k <= 0 || a_nnz <= 0 || size_t(a_nnz) > size_t(m)*k || repeat <= 0) return 1;

  const int transpose = 1;
  const int A_num_rows = m;
  const int A_num_cols = k;
  const int A_nnz = a_nnz;
  const int B_nnz = a_nnz;
  const int B_num_rows = A_num_cols;
  const int B_num_cols = A_num_rows;
  const int lda = A_num_cols;
  const int ldb = B_num_cols;
  const int A_size = lda * A_num_rows;
  const int B_size = ldb * B_num_rows;

  float* hA = static_cast<float*>(std::malloc(A_size * sizeof(float)));
  const size_t A_value_size_bytes = A_nnz * sizeof(float);
  const size_t A_colidx_size_bytes = A_nnz * sizeof(int);
  const size_t A_rowidx_size_bytes = (A_num_rows + 1) * sizeof(int);

  float* hA_values = static_cast<float*>(std::malloc(A_value_size_bytes));
  int* hA_columns = static_cast<int*>(std::malloc(A_colidx_size_bytes));
  int* hA_offsets = static_cast<int*>(std::malloc(A_rowidx_size_bytes));

  init_matrix(hA, A_num_rows, A_num_cols, A_nnz);
  init_csr(hA_offsets, hA_values, hA_columns, hA, A_num_rows, A_num_cols, A_nnz);

  // IRIS initialization can consume process RNG state: retain original inputs.
  iris::Platform platform;
  if (platform.init(nullptr, nullptr, 1) != IRIS_SUCCESS) return 2;
  int backend=-1; char device_name[256]={};
  iris_device_info(0,iris_backend,&backend,nullptr);
  iris_device_info(0,iris_name,device_name,nullptr);
  if(backend!=iris_hip) return 2;
  std::printf("IRIS HIP backend=%d device=%s\n",backend,device_name);

  const size_t B_value_size_bytes = B_nnz * sizeof(float);
  const size_t B_rowidx_size_bytes = B_nnz * sizeof(int);
  const size_t B_colidx_size_bytes = (A_num_cols + 1) * sizeof(int);

  float* hB_values_tmp = static_cast<float*>(std::malloc(B_value_size_bytes));
  int* hB_rows_tmp = static_cast<int*>(std::malloc(B_rowidx_size_bytes));
  int* hB_offsets_tmp = static_cast<int*>(std::malloc(B_colidx_size_bytes));
  int* hNext = static_cast<int*>(std::malloc(B_colidx_size_bytes));

  iris::DMem mem_A_offsets(hA_offsets, A_rowidx_size_bytes);
  iris::DMem mem_A_columns(hA_columns, A_colidx_size_bytes);
  iris::DMem mem_A_values(hA_values, A_value_size_bytes);
  iris::DMem mem_B_offsets(hB_offsets_tmp, B_colidx_size_bytes);
  iris::DMem mem_B_rows(hB_rows_tmp, B_rowidx_size_bytes);
  iris::DMem mem_B_values(hB_values_tmp, B_value_size_bytes);
  iris::DMem mem_next(hNext, B_colidx_size_bytes);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_csr2csc(mem_A_offsets, mem_A_columns, mem_A_values, mem_B_offsets, mem_B_rows, mem_B_values,
                   mem_next, A_num_rows, A_num_cols, A_nnz);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of IRIS csr2csc : %f (us)\n", (time * 1e-3f) / repeat);

  if (verify) {
    std::printf("Computing the reference results..\n");
    float* hB = static_cast<float*>(std::malloc(B_size * sizeof(float)));
    transpose_dense(hA, hB, A_num_cols, A_num_rows, transpose);

    float* hB_values = static_cast<float*>(std::malloc(B_value_size_bytes));
    int* hB_rows = static_cast<int*>(std::malloc(B_rowidx_size_bytes));
    int* hB_offsets = static_cast<int*>(std::malloc(B_colidx_size_bytes));

    init_csr(hB_offsets, hB_values, hB_rows, hB, B_num_rows, B_num_cols, B_nnz);

    int correct = 1;
    for (int i = 0; i < B_num_rows + 1; i++) {
      if (hB_offsets_tmp[i] != hB_offsets[i]) {
        correct = 0;
        break;
      }
    }
    for (int i = 0; i < B_nnz; i++) {
      if (hB_rows_tmp[i] != hB_rows[i] || std::fabs(hB_values_tmp[i] - hB_values[i]) > 1e-2f) {
        correct = 0;
        break;
      }
    }

    if (const char* prefix = std::getenv("HECBENCH_DUMP_PREFIX")) {
      char name[4096];
      std::snprintf(name, sizeof(name), "%s.bin", prefix);
      FILE* f = std::fopen(name, "wb");
      if (!f) return 3;
      std::fwrite(hB_offsets_tmp, sizeof(int), B_num_rows+1, f);
      std::fwrite(hB_rows_tmp, sizeof(int), B_nnz, f);
      std::fwrite(hB_values_tmp, sizeof(float), B_nnz, f);
      std::fclose(f);
    }

    std::free(hB);
    std::free(hB_values);
    std::free(hB_rows);
    std::free(hB_offsets);

    if (correct) {
      std::printf("spgeam_example test PASSED\n");
    } else {
      std::printf("spgeam_example test FAILED: wrong result\n");
    }
  }

  std::free(hA);
  std::free(hA_values);
  std::free(hA_columns);
  std::free(hA_offsets);
  std::free(hB_values_tmp);
  std::free(hB_rows_tmp);
  std::free(hB_offsets_tmp);
  std::free(hNext);
  platform.finalize();
  return EXIT_SUCCESS;
}
