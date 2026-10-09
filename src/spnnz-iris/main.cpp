#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void init_matrix(float* matrix, int64_t num_rows, int64_t num_cols,
                        int64_t nnz) {
  const int64_t n = num_rows * num_cols;
  float* d = static_cast<float*>(std::malloc(n * sizeof(float)));

  std::srand(123);
  for (int64_t i = 0; i < n; i++) d[i] = static_cast<float>(i);
  for (int64_t i = n; i > 0; i--) {
    const int64_t a = i - 1;
    const int64_t b = std::rand() % i;
    if (a != b) {
      const float t = d[a];
      d[a] = d[b];
      d[b] = t;
    }
  }

  srand48(123);
  for (int64_t i = 0; i < num_rows; i++) {
    for (int64_t j = 0; j < num_cols; j++) {
      matrix[i * num_cols + j] =
          (d[i * num_cols + j] >= nnz) ? 0.0f : static_cast<float>(drand48() + 1);
    }
  }

  std::free(d);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::printf("This function computes the number of nonzero elements per row");
    std::printf(" and the total number of nonzero elements in a dense matrix.\n");
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

  float* h_dense =
      static_cast<float*>(std::malloc(dense_size * sizeof(float)));
  int* nnz_per_row =
      static_cast<int*>(std::malloc(num_rows * sizeof(int)));
  if (!h_dense || !nnz_per_row) {
    std::fprintf(stderr, "host allocation failed\n");
    return 1;
  }

  std::printf("Initializing host matrices..\n");
  init_matrix(h_dense, num_rows, num_cols, h_nnz);

  iris::DMem mem_dense(h_dense, dense_size * sizeof(float));
  iris::DMem mem_row_counts(nnz_per_row, num_rows * sizeof(int));

  size_t lws = 256;
  size_t gws = ((static_cast<size_t>(num_rows) + lws - 1) / lws) * lws;
  void* params[] = {&mem_row_counts, &mem_dense, (void*)&num_rows,
                    (void*)&num_cols};
  int pinfo[] = {iris_w, iris_r, sizeof(num_rows), sizeof(num_cols)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    task.kernel("spnnz_count_rows", 1, nullptr, &gws, &lws, 4, params, pinfo);
    task.flush_out(mem_row_counts);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of iris_spnnz : %f (us)\n",
              (time * 1e-3f) / repeat);

  int correct = 1;
  int nnz_total = 0;
  for (int64_t i = 0; i < num_rows; i++) nnz_total += nnz_per_row[i];
  if (h_nnz != nnz_total) {
    std::printf("nnz: %d != %d\n", static_cast<int>(h_nnz), nnz_total);
    correct = 0;
    goto print_error;
  }

  for (int64_t i = 0; i < num_rows; i++) {
    int nnz = 0;
    for (int64_t j = 0; j < num_cols; j++) {
      if (h_dense[i * num_cols + j] != 0.0f) nnz++;
    }
    if (nnz != nnz_per_row[i]) {
      std::printf("@row %ld %d != %d\n", i, nnz, nnz_per_row[i]);
      correct = 0;
      goto print_error;
    }
  }

print_error:
  if (correct)
    std::printf("sparse_nnz_example test PASSED\n");
  else
    std::printf("sparse_nnz_example test FAILED: wrong result\n");

  std::free(h_dense);
  std::free(nnz_per_row);
  platform.finalize();
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
