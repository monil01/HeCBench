#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "iris_dmem.hpp"

#define CUDA_CHECK(call)                                                        \
  do {                                                                         \
    int err__ = (call);                                               \
    if (err__ != IRIS_SUCCESS) {                                               \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
              __LINE__, iris_error_string(err__));                           \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Full Jacobian of a batched softmax.
//
// For a single sample x in R^D:
//     y = softmax(x),   y_i = exp(x_i - max) / sum_k exp(x_k - max)
// the (full) Jacobian J = dy/dx in R^{D x D} has the closed form
//     J[i][j] = dy_i / dx_j = y_i * (delta_ij - y_j)
//
// This is exactly the local Jacobian used by the backward pass of softmax:
// the reverse-mode gradient (VJP) is  dL/dx = J^T (dL/dy), and here we
// materialize the *entire* J for every sample in the batch (output B x D x D).
//
// The results are verified against a CPU reference here, and against PyTorch's
// reverse-mode (vjp / jacrev) and forward-mode (jvp / jacfwd) autodiff in
// verify_torch.py, which reads the small dump written by dump_reference().
// ---------------------------------------------------------------------------

template <typename T>
void reference(const T* input, T* jac, int batch_size, int dim)
{
  for (int b = 0; b < batch_size; b++) {
    const T* x = input + (size_t)b * dim;
    T* J = jac + (size_t)b * dim * dim;

    T maxv = x[0];
    #pragma omp parallel for reduction(max:maxv)
    for (int i = 1; i < dim; i++) maxv = std::max(x[i], maxv);

    T sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < dim; i++) sum += std::exp(x[i] - maxv);
    const T inv = (T)1 / sum;

    // reuse the front of J as scratch for y, then overwrite
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        const T yi = std::exp(x[i] - maxv) * inv;
        const T yj = std::exp(x[j] - maxv) * inv;
        J[(size_t)i * dim + j] = yi * ((i == j ? (T)1 : (T)0) - yj);
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  IRISSession session(&argc,&argv);
  const int batch_size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  assert(batch_size > 0);

  printf("Batch size: %d\n", batch_size);

  const int dims[] = {128, 512, 2048, 8192};

  for (size_t n = 0; n < sizeof(dims) / sizeof(int); n++) {
    const int dim = dims[n];
    printf("\nSoftmax dimension: %d (Jacobian %d x %d per sample)\n", dim, dim, dim);

    const size_t in_elems  = (size_t)batch_size * dim;
    const size_t jac_elems = (size_t)batch_size * dim * dim;
    const size_t in_bytes  = in_elems * sizeof(float);
    const size_t jac_bytes = jac_elems * sizeof(float);

    float* input      = (float*)malloc(in_bytes);
    float* jac_k1     = (float*)malloc(jac_bytes);
    float* jac_k2     = (float*)malloc(jac_bytes);
    float* jac_ref    = (float*)malloc(jac_bytes);

    std::default_random_engine g(123);
    std::uniform_real_distribution<float> distr(-3.f, 3.f);
    for (size_t i = 0; i < in_elems; i++) input[i] = distr(g);

    reference(input, jac_ref, batch_size, dim);

    float *d_input, *d_jac;
    CUDA_CHECK(iris_alloc((void**)&d_input, in_bytes));
    CUDA_CHECK(iris_alloc((void**)&d_jac, jac_bytes));
    CUDA_CHECK(iris_copy(d_input, input, in_bytes, IRISHostToDevice));

    for (int block_size = 64; block_size <= 1024; block_size *= 2) {
      printf("block size: %d\n", block_size);
      const size_t shmem = ((size_t)dim + block_size) * sizeof(float);

      CUDA_CHECK(iris_set(d_jac, 0, jac_bytes));
      iris_launch("jacobian_k1", batch_size, block_size, d_input, d_jac, dim);
      CUDA_CHECK(iris_copy(jac_k1, d_jac, jac_bytes, IRISDeviceToHost));

      CUDA_CHECK(iris_set(d_jac, 0, jac_bytes));
      iris_launch("jacobian_k2", batch_size, block_size, d_input, d_jac, dim);
      CUDA_CHECK(iris_copy(jac_k2, d_jac, jac_bytes, IRISDeviceToHost));

      bool ok = true;
      for (size_t i = 0; i < jac_elems; i++) {
        if (fabsf(jac_k1[i] - jac_ref[i]) > 1e-3f ||
            fabsf(jac_k2[i] - jac_ref[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }

    printf("Benchmarking..\n");

    for (int block_size = 64; block_size <= 1024; block_size *= 2) {
      printf("block size: %d\n", block_size);
      const size_t shmem = ((size_t)dim + block_size) * sizeof(float);

      CUDA_CHECK(iris_set(d_jac, 0, jac_bytes));
      CUDA_CHECK(iris_sync());
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        iris_launch("jacobian_k1", batch_size, block_size, d_input, d_jac, dim);
      CUDA_CHECK(iris_sync());
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of softmax Jacobian kernel (k1): %f (us)\n",
             (time * 1e-3f) / repeat);

      CUDA_CHECK(iris_set(d_jac, 0, jac_bytes));
      CUDA_CHECK(iris_sync());
      start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        iris_launch("jacobian_k2", batch_size, block_size, d_input, d_jac, dim);
      CUDA_CHECK(iris_sync());
      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of softmax Jacobian kernel (k2): %f (us)\n",
             (time * 1e-3f) / repeat);
    }

    CUDA_CHECK(iris_free(d_input));
    CUDA_CHECK(iris_free(d_jac));
    free(input);
    free(jac_k1);
    free(jac_k2);
    free(jac_ref);
  }

  return 0;
}
