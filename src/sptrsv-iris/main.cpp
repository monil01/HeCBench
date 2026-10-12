#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../sptrsv-sycl/mmio.h"
#include "../sptrsv-sycl/sptrsv.h"
#include "../sptrsv-sycl/tranpose.h"
#include "../sptrsv-sycl/read_mtx.h"

int sptrsv_syncfree_iris(const int repeat,
                         const int* csrRowPtr,
                         const int* csrColIdx,
                         const VALUE_TYPE* csrVal,
                         const int m,
                         const int n,
                         const int nnz,
                         VALUE_TYPE* x,
                         const VALUE_TYPE* b,
                         const VALUE_TYPE* x_ref)
{
  if (m != n) {
    std::printf("This is not a square matrix, return.\n");
    return -1;
  }

  int* warp_num = static_cast<int*>(std::malloc((m + 1) * sizeof(int)));
  std::memset(warp_num, 0, sizeof(int) * (m + 1));
  std::memset(x, 0, m * sizeof(VALUE_TYPE));

  double warp_occupy = 0.0;
  double element_occupy = 0.0;
  int Len = 0;
  for (int i = 0; i < repeat; i++) {
    matrix_warp4(m, n, nnz, csrRowPtr, csrColIdx, csrVal, 10, &Len, warp_num,
                 &warp_occupy, &element_occupy);
  }

  int* get_value = static_cast<int*>(std::malloc(sizeof(int) * m));
  const int num_threads = WARP_PER_BLOCK * WARP_SIZE;
  const int num_blocks = static_cast<int>(
      std::ceil(static_cast<double>((Len - 1) * WARP_SIZE) /
                static_cast<double>(num_threads)));
  const size_t gws[1] = { static_cast<size_t>(num_blocks * num_threads) };
  const size_t lws[1] = { static_cast<size_t>(num_threads) };

  iris::DMem mem_csrRowPtr(const_cast<int*>(csrRowPtr), sizeof(int) * (n + 1));
  iris::DMem mem_csrColIdx(const_cast<int*>(csrColIdx), sizeof(int) * nnz);
  iris::DMem mem_csrVal(const_cast<VALUE_TYPE*>(csrVal), sizeof(VALUE_TYPE) * nnz);
  iris::DMem mem_get_value(get_value, sizeof(int) * m);
  iris::DMem mem_b(const_cast<VALUE_TYPE*>(b), sizeof(VALUE_TYPE) * m);
  iris::DMem mem_x(x, sizeof(VALUE_TYPE) * n);
  iris::DMem mem_warp_num(warp_num, sizeof(int) * Len);

  double time = 0.0;
  for (int i = 0; i <= repeat; i++) {
    iris::Task init_task;
    void* init_params[] = { &mem_get_value, &mem_x, const_cast<int*>(&m), const_cast<int*>(&n) };
    int init_pinfo[] = { iris_w, iris_w, sizeof(m), sizeof(n) };
    const size_t init_gws[1] = { static_cast<size_t>(std::max(m, n)) };
    const size_t init_lws[1] = { 256 };
    init_task.kernel("sptrsv_init_buffers", 1, nullptr, const_cast<size_t*>(init_gws),
                     const_cast<size_t*>(init_lws), 4, init_params, init_pinfo);
    init_task.submit(iris_gpu, nullptr, 1);

    iris::Task task;
    void* params[] = {
        &mem_csrRowPtr, &mem_csrColIdx, &mem_csrVal, &mem_get_value,
        const_cast<int*>(&m), &mem_b, &mem_x, &mem_warp_num, const_cast<int*>(&Len)};
    int pinfo[] = {
        iris_r, iris_r, iris_r, iris_rw, sizeof(m), iris_r, iris_rw, iris_r, sizeof(Len)};

    auto start = std::chrono::steady_clock::now();
    task.kernel("sptrsv_mix", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 9, params, pinfo);
    task.flush_out(mem_x);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();
    if (i > 0) {
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
  }

  std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  const double accuracy = 1e-4;
  double ref = 0.0;
  double res = 0.0;
  int error = 0;
  const int rhs = 1;
  for (int i = 0; i < n * rhs; i++) {
    ref += std::abs(x_ref[i]);
    res += std::abs(x[i] - x_ref[i]);
    if (x[i] != x_ref[i] && error < 10) {
      error++;
    }
  }
  res = ref == 0.0 ? res : res / ref;

  std::printf("|x-xref|/|xref| = %8.2e\n", res);
  std::printf("%s\n", (res < accuracy) ? "PASS" : "FAIL");

  std::free(get_value);
  std::free(warp_num);
  return (res < accuracy) ? 0 : 1;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <example.mtx> <repeat>\n", argv[0]);
    return -1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  char* filename = argv[1];
  const int repeat = std::atoi(argv[2]);

  int m = 0;
  int n = 0;
  int nnzA = 0;
  int* csrRowPtrA = nullptr;
  int* csrColIdxA = nullptr;
  VALUE_TYPE* csrValA = nullptr;
  read_mtx(filename, &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);

  int nnzL = 0;
  int* csrRowPtrL_tmp = nullptr;
  int* csrColIdxL_tmp = nullptr;
  VALUE_TYPE* csrValL_tmp = nullptr;

  if (m <= n) {
    n = m;
  } else {
    m = n;
  }
  if (m <= 1) {
    platform.finalize();
    return 0;
  }

  change2tran(m, nnzA, csrRowPtrA, csrColIdxA, csrValA, &nnzL,
              &csrRowPtrL_tmp, &csrColIdxL_tmp, &csrValL_tmp);

  std::free(csrColIdxA);
  std::free(csrValA);
  std::free(csrRowPtrA);

  if (m == 0 || nnzL == 0) {
    platform.finalize();
    return -3;
  }

  int layer = 0;
  double parallelism = 0.0;
  matrix_layer(m, n, nnzL, csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp,
               &layer, &parallelism);

  VALUE_TYPE* x_ref = nullptr;
  VALUE_TYPE* b = nullptr;
  get_x_b(m, n, csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, &x_ref, &b);
  VALUE_TYPE* x = static_cast<VALUE_TYPE*>(std::malloc(sizeof(VALUE_TYPE) * m));

  const int rc = sptrsv_syncfree_iris(repeat, csrRowPtrL_tmp, csrColIdxL_tmp,
                                      csrValL_tmp, m, n, nnzL, x, b, x_ref);

  std::free(csrColIdxL_tmp);
  std::free(csrValL_tmp);
  std::free(csrRowPtrL_tmp);
  std::free(x_ref);
  std::free(b);
  std::free(x);

  platform.finalize();
  return rc;
}
