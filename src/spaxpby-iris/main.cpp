#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define BLOCK_SIZE 256

static int init_sparse_from_cuda_pattern(
    size_t size,
    size_t nnz,
    size_t* indices,
    float* values)
{
  float* d = static_cast<float*>(std::malloc(size * sizeof(float)));
  if (!d) return 1;

  std::srand(123);
  for (size_t i = 0; i < size; i++) d[i] = static_cast<float>(i);
  for (size_t i = size; i > 0; i--) {
    const size_t a = i - 1;
    const size_t b = static_cast<size_t>(std::rand()) % i;
    if (a != b) {
      const float t = d[a];
      d[a] = d[b];
      d[b] = t;
    }
  }

  ::srand48(123);
  size_t k = 0;
  for (size_t i = 0; i < size; i++) {
    if (d[i] < nnz) {
      if (k >= nnz) {
        std::free(d);
        return 2;
      }
      indices[k] = i;
      values[k] = static_cast<float>(::drand48() + 1.0);
      k++;
    }
  }

  std::free(d);
  return k == nnz ? 0 : 3;
}

static void init_dense_from_cuda_pattern(float* y, size_t size)
{
  ::srand48(123);
  for (size_t i = 0; i < size; i++) {
    y[i] = static_cast<float>(::drand48() + 1.0);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("The function computes the sum of a sparse vector and a dense vector ");
    std::printf("in single-precision floating-point operations\n");
    std::printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const size_t m = static_cast<size_t>(std::atol(argv[1]));
  const size_t n = static_cast<size_t>(std::atol(argv[2]));
  const size_t nnz = static_cast<size_t>(std::atol(argv[3]));
  const int repeat = std::atoi(argv[4]);
  const size_t size = m * n;

  const size_t value_size_bytes = nnz * sizeof(float);
  const size_t index_size_bytes = nnz * sizeof(size_t);
  const size_t dense_size_bytes = size * sizeof(float);

  float* y = static_cast<float*>(std::malloc(dense_size_bytes));
  float* y_ref = static_cast<float*>(std::malloc(dense_size_bytes));
  float* values = static_cast<float*>(std::malloc(value_size_bytes));
  size_t* indices = static_cast<size_t*>(std::malloc(index_size_bytes));
  if (!y || !y_ref || !values || !indices) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(y);
    std::free(y_ref);
    std::free(values);
    std::free(indices);
    platform.finalize();
    return 1;
  }

  std::printf("Initializing input matrices..\n");
  if (init_sparse_from_cuda_pattern(size, nnz, indices, values) != 0) {
    std::fprintf(stderr, "sparse initialization failed\n");
    std::free(y);
    std::free(y_ref);
    std::free(values);
    std::free(indices);
    platform.finalize();
    return 1;
  }
  init_dense_from_cuda_pattern(y, size);
  for (size_t i = 0; i < size; i++) y_ref[i] = y[i];
  std::printf("Done\n");

  const float alpha = 1.0f;
  const float beta = 1.0f;
  (void)beta;
  const float alpha_repeat = alpha * static_cast<float>(repeat);

  iris::DMem mem_indices(indices, index_size_bytes);
  iris::DMem mem_values(values, value_size_bytes);
  iris::DMem mem_y(y, dense_size_bytes);

  const size_t blocks = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  iris::Task task;
  void* params[5] = {
    const_cast<size_t*>(&nnz),
    const_cast<float*>(&alpha_repeat),
    &mem_indices,
    &mem_values,
    &mem_y
  };
  int pinfo[5] = {
    sizeof(nnz),
    sizeof(alpha_repeat),
    iris_r,
    iris_r,
    iris_rw
  };
  task.kernel("spaxpby_kernel", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
  task.flush_out(mem_y);
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of SPAXPBY : %f (us)\n", (time * 1e-3f) / repeat);

  mem_y.fetch(y);

  std::printf("Computing the reference results..\n");
  for (size_t k = 0; k < nnz; k++) {
    y_ref[indices[k]] += alpha_repeat * values[k];
  }
  std::printf("Done\n");

  int correct = 1;
  for (size_t i = 0; i < size; i++) {
    if (std::fabs(y[i] - y_ref[i]) > 1e-2f) {
      correct = 0;
      break;
    }
  }
  if (correct)
    std::printf("axpby_example test PASSED\n");
  else
    std::printf("axpby_example test FAILED: wrong result\n");

  std::free(y);
  std::free(y_ref);
  std::free(values);
  std::free(indices);
  platform.finalize();
  return correct ? 0 : 1;
}
