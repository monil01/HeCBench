#include <iris/iris.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "../spsort-sycl/utils.h"

template <typename fp, typename intType>
static int sort_sparse_matrix(intType a_nrows, intType a_ncols, intType a_nnz, int repeat)
{
  fp* m = static_cast<fp*>(std::malloc(sizeof(fp) * a_nrows * a_ncols));
  init_matrix(m, a_nrows, a_ncols, a_nnz);

  intType* ia = static_cast<intType*>(std::malloc((a_nrows + 1) * sizeof(intType)));
  intType* ja = static_cast<intType*>(std::malloc(a_nnz * sizeof(intType)));
  fp* a = static_cast<fp*>(std::malloc(a_nnz * sizeof(fp)));
  init_csr(ia, a, ja, m, a_nrows, a_ncols, a_nnz);

  intType* ia_result = static_cast<intType*>(std::malloc((a_nrows + 1) * sizeof(intType)));
  intType* ja_result = static_cast<intType*>(std::malloc(a_nnz * sizeof(intType)));
  fp* a_result = static_cast<fp*>(std::malloc(a_nnz * sizeof(fp)));
  if (!m || !ia || !ja || !a || !ia_result || !ja_result || !a_result) {
    std::cerr << "host allocation failed" << std::endl;
    return EXIT_FAILURE;
  }

  std::memcpy(ia_result, ia, (a_nrows + 1) * sizeof(intType));
  std::memcpy(ja_result, ja, a_nnz * sizeof(intType));
  std::memcpy(a_result, a, a_nnz * sizeof(fp));
  shuffle_matrix_data<fp, intType>(ia_result, ja_result, a_result, a_nrows, a_nnz);

  std::cout << "\n\t\tBasic info of the sparse matrix:\n";
  const fp ave_nnz_per_row = static_cast<fp>(a_nnz) / a_nrows;
  std::cout << "\t\t\tA_nrows = A_ncols = " << a_nrows << std::endl;
  std::cout << "\t\t\tA_nnz   = " << a_nnz << std::endl;

  intType max_nnz_per_row = 0;
  intType min_nnz_per_row = a_nnz;
  for (intType row = 0; row < a_nrows; ++row) {
    const intType loc_nnz_per_row = ia[row + 1] - ia[row];
    if (loc_nnz_per_row > max_nnz_per_row) max_nnz_per_row = loc_nnz_per_row;
    if (loc_nnz_per_row < min_nnz_per_row) min_nnz_per_row = loc_nnz_per_row;
  }

  std::cout << "\t\t\t\tmin_nnz_per_row = " << min_nnz_per_row << std::endl;
  std::cout << "\t\t\t\tave_nnz_per_row = " << ave_nnz_per_row << std::endl;
  std::cout << "\t\t\t\tmax_nnz_per_row = " << max_nnz_per_row << std::endl;
  std::cout << std::endl;

  iris::DMem rowptr(ia_result, (a_nrows + 1) * sizeof(intType));
  iris::DMem colind(ja_result, a_nnz * sizeof(intType));
  iris::DMem values(a_result, a_nnz * sizeof(fp));

  const size_t gws[1] = {static_cast<size_t>(a_nrows)};
  const size_t lws[1] = {1};
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {&rowptr, &colind, &values, &a_nrows};
    int pinfo[4] = {iris_r, iris_rw, iris_rw, sizeof(a_nrows)};
    task.kernel("csr_sort_rows_float_int", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 4, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(colind);
      task.flush_out(values);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of CSR sort: " << (time * 1e-3f) / repeat
            << " us" << std::endl;

  int error = std::memcmp(ia_result, ia, (a_nrows + 1) * sizeof(intType));
  if (error) {
    std::cout << "Error: row index arrays mismatch" << std::endl;
    goto print_error;
  }
  error = std::memcmp(ja_result, ja, a_nnz * sizeof(intType));
  if (error) {
    std::cout << "Error: column index arrays mismatch" << std::endl;
    goto print_error;
  }
  error = std::memcmp(a_result, a, a_nnz * sizeof(fp));
  if (error) {
    std::cout << "Error: value arrays mismatch" << std::endl;
    goto print_error;
  }

print_error:
  if (error == 0)
    std::cout << "csrsort_example test PASSED" << std::endl;
  else
    std::cout << "csrsort_example test FAILED: wrong result" << std::endl;

  std::free(m);
  std::free(ia);
  std::free(ja);
  std::free(a);
  std::free(ia_result);
  std::free(ja_result);
  std::free(a_result);
  return error == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::cout << "The function sorts the column indices of each row in a sparse MxN matrix"
              << std::endl;
    std::cout << "The sparse matrix is represented in CSR (Compressed Sparse Row) storage format"
              << std::endl;
    std::cout << "Usage " << argv[0] << " <M> <N> <nnz> <repeat>" << std::endl;
    std::cout << "nnz is the number of non-zero elements" << std::endl;
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  const int m = std::atoi(argv[1]);
  const int n = std::atoi(argv[2]);
  const int nnz = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);
  int status = sort_sparse_matrix<float, int>(m, n, nnz, repeat);
  platform.finalize();
  return status;
}
