#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

#define MAX_KERNEL_THREADS 256

typedef float vtype;
typedef vector<vector<vtype>> matrix;

static void run_kernel(const char* name, int dim, size_t* gws, size_t* lws,
                       int nparams, void** params, int* pinfo,
                       iris::DMem* flush = nullptr)
{
  iris::Task task;
  task.kernel(name, dim, nullptr, gws, lws, nparams, params, pinfo);
  if (flush) task.flush_out(*flush);
  task.submit(iris_gpu, nullptr, 1);
}

static void jaccard_weight(bool weighted, const int iteration, const int n, const int e,
                           int* csr_ptr, int* csr_ind, vtype* csr_val)
{
  const vtype gamma = static_cast<vtype>(0.46);

#ifdef DEBUG
  vector<vtype> weight_i(e);
  vector<vtype> weight_s(e);
  vector<vtype> work(n);
#endif
  vector<vtype> weight_j(e);
  vector<vtype> work_device(n);
  vector<vtype> weight_i_device(e);
  vector<vtype> weight_s_device(e);

  iris::DMem d_work(work_device.data(), sizeof(vtype) * n);
  iris::DMem d_weight_i(weight_i_device.data(), sizeof(vtype) * e);
  iris::DMem d_weight_s(weight_s_device.data(), sizeof(vtype) * e);
  iris::DMem d_weight_j(weight_j.data(), sizeof(vtype) * e);
  iris::DMem d_csr_val(csr_val, sizeof(vtype) * e);
  iris::DMem d_csr_ptr(csr_ptr, sizeof(int) * (n + 1));
  iris::DMem d_csr_ind(csr_ind, sizeof(int) * e);

  auto start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < iteration; iter++) {
    int threads_x = MAX_KERNEL_THREADS;
    size_t fill_gws[1] = {static_cast<size_t>(((e + threads_x - 1) / threads_x) * threads_x)};
    size_t fill_lws[1] = {static_cast<size_t>(threads_x)};
    vtype one = static_cast<vtype>(1.0);
    void* fill_j_params[3] = {&d_weight_j, const_cast<int*>(&e), &one};
    int fill_j_pinfo[3] = {iris_w, sizeof(e), sizeof(one)};
    run_kernel(weighted ? "jaccard_fill_true" : "jaccard_fill_false",
               1, fill_gws, fill_lws, 3, fill_j_params, fill_j_pinfo);

    vtype zero = static_cast<vtype>(0.0);
    void* fill_i_params[3] = {&d_weight_i, const_cast<int*>(&e), &zero};
    int fill_i_pinfo[3] = {iris_w, sizeof(e), sizeof(zero)};
    run_kernel("jaccard_fill_false", 1, fill_gws, fill_lws, 3, fill_i_params, fill_i_pinfo);

    const int y = 4;
    size_t row_lws[2] = {static_cast<size_t>(64 / y), static_cast<size_t>(y)};
    size_t row_gws[2] = {
      row_lws[0],
      static_cast<size_t>(((n + static_cast<int>(row_lws[1]) - 1) / static_cast<int>(row_lws[1])) * static_cast<int>(row_lws[1]))
    };
    void* row_params[5] = {const_cast<int*>(&n), &d_csr_ptr, &d_csr_ind, &d_weight_j, &d_work};
    int row_pinfo[5] = {sizeof(n), iris_r, iris_r, iris_r, iris_w};
    run_kernel(weighted ? "jaccard_row_sum_true" : "jaccard_row_sum_false",
               2, row_gws, row_lws, 5, row_params, row_pinfo);

    size_t is_lws[3] = {static_cast<size_t>(32 / y), static_cast<size_t>(y), 8};
    size_t is_gws[3] = {
      is_lws[0],
      is_lws[1],
      static_cast<size_t>(((n + static_cast<int>(is_lws[2]) - 1) / static_cast<int>(is_lws[2])) * static_cast<int>(is_lws[2]))
    };
    void* is_params[8] = {
      const_cast<int*>(&n), const_cast<int*>(&e), &d_csr_ptr, &d_csr_ind,
      &d_weight_j, &d_work, &d_weight_i, &d_weight_s
    };
    int is_pinfo[8] = {sizeof(n), sizeof(e), iris_r, iris_r, iris_r, iris_r, iris_rw, iris_w};
    run_kernel(weighted ? "jaccard_is_opt_true" : "jaccard_is_opt_false",
               3, is_gws, is_lws, 8, is_params, is_pinfo);

    threads_x = std::min(e, MAX_KERNEL_THREADS);
    size_t jw_lws[1] = {static_cast<size_t>(threads_x)};
    size_t jw_gws[1] = {static_cast<size_t>(((e + threads_x - 1) / threads_x) * threads_x)};
    void* jw_params[6] = {const_cast<int*>(&e), &d_csr_val, const_cast<vtype*>(&gamma),
                          &d_weight_i, &d_weight_s, &d_weight_j};
    int jw_pinfo[6] = {sizeof(e), iris_r, sizeof(gamma), iris_r, iris_r, iris_w};
    run_kernel("jaccard_jw", 1, jw_gws, jw_lws, 6, jw_params, jw_pinfo,
               iter == iteration - 1 ? &d_weight_j : nullptr);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cout << "Average execution time of kernels: " << (time * 1e-9f) / iteration << " (s)\n";

  d_weight_j.fetch(weight_j.data());

#ifdef DEBUG
  float error;
  if (weighted)
    error = std::fabs(weight_j[0] - 0.306667f) +
            std::fabs(weight_j[1] - 0.000000f) +
            std::fabs(weight_j[2] - 3.680000f) +
            std::fabs(weight_j[3] - 1.380000f) +
            std::fabs(weight_j[4] - 0.788571f) +
            std::fabs(weight_j[5] - 0.460000f);
  else
    error = std::fabs(weight_j[0] - 0.230000f) +
            std::fabs(weight_j[1] - 0.000000f) +
            std::fabs(weight_j[2] - 3.680000f) +
            std::fabs(weight_j[3] - 1.380000f) +
            std::fabs(weight_j[4] - 0.920000f) +
            std::fabs(weight_j[5] - 0.460000f);

  if (error > 1e-5f) {
    for (int i = 0; i < e; i++) printf("wj: %d %f\n", i, weight_j[i]);
    printf("FAILED");
  } else {
    printf("PASSED");
  }
  printf("\n");
#endif
}

static void printMatrix(const matrix& M)
{
  int m = M.size();
  int n = M[0].size();
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
      cout << M[i][j] << " ";
    cout << endl;
  }
}

template <typename T>
static void printVector(const vector<T>& V, char* msg)
{
  cout << msg << "[ ";
  for_each(V.begin(), V.end(), [](int a) { cout << a << " "; });
  cout << "]" << endl;
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int iteration = 10;

#ifdef DEBUG
  matrix M = {
    { 0, 0, 0, 1},
    { 5, 8, 0, 0},
    { 0, 0, 3, 0},
    { 0, 6, 0, 1}
  };
#else
  if (argc != 4) {
    printf("Usage: %s <num rows> <num cols> <iterations>\n", argv[0]);
    platform.finalize();
    return 1;
  }

  int numRow = atoi(argv[1]);
  int numCol = atoi(argv[2]);
  iteration = atoi(argv[3]);

  srand(2);
  matrix M;
  vector<vtype> rowElems(numCol);
  for (int r = 0; r < numRow; r++) {
    for (int c = 0; c < numCol; c++)
      rowElems[c] = rand() % 10;
    M.push_back(rowElems);
  }
#endif

  int row = M.size();
  int col = M[0].size();
  printf("Number of matrix rows and cols: %d %d\n", row, col);
  vector<vtype> csr_val;
  vector<int> csr_ptr = {0};
  vector<int> csr_ind;
  int nnz = 0;

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (M[i][j] != static_cast<vtype>(0)) {
        csr_val.push_back(M[i][j]);
        csr_ind.push_back(j);
        nnz++;
      }
    }
    csr_ptr.push_back(nnz);
  }

  if (row <= 16 && col <= 16) {
    printMatrix(M);
    printVector(csr_val, (char*)"values = ");
    printVector(csr_ptr, (char*)"row pointer = ");
    printVector(csr_ind, (char*)"col indices = ");
  }

  jaccard_weight(true, iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());
  jaccard_weight(false, iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());

  platform.finalize();
  return 0;
}
