#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <unistd.h>

#include "reference.h"

template <typename T>
static void gemmBatched(int lower, int upper, int num, int reps, int verbose, const char* kernel_name)
{
  if (verbose) {
    std::cout << "initializing inputs" << std::endl;
  }
  const size_t matrices_size = static_cast<size_t>(upper) * upper * num * sizeof(T);
  const size_t vectors_size = static_cast<size_t>(upper) * num * sizeof(T);
  T* matrices = static_cast<T*>(std::malloc(matrices_size));
  T* vectors = static_cast<T*>(std::malloc(vectors_size));
  T* result = static_cast<T*>(std::malloc(vectors_size));
  T* result_ref = static_cast<T*>(std::malloc(vectors_size));
  assert(matrices && vectors && result && result_ref);

  srand48(48);
  for (int i = 0; i < num * upper * upper; i++) {
    matrices[i] = static_cast<T>(drand48());
  }
  for (int i = 0; i < num * upper; i++) {
    vectors[i] = static_cast<T>(drand48());
    result[i] = T(0);
    result_ref[i] = T(0);
  }

  iris::DMem mem_matrices(matrices, matrices_size);
  iris::DMem mem_vectors(vectors, vectors_size);
  iris::DMem mem_result(result, vectors_size);

  const int lda = upper;
  const int ldb = upper;
  const int ldc = upper;
  const T alpha = T(1.0);
  const T beta = T(0.0);

  for (int size = lower; size <= upper; size++) {
    if (verbose) {
      std::cout << "running with <size x size> x <size x 1> " << size << std::endl;
    }
    double sum = 0.0;
    const int m = size;
    const int n = 1;
    const int k = size;
    const size_t lws[1] = {256};
    const size_t gws[1] = {static_cast<size_t>((num * m + 255) / 256 * 256)};

    for (int rep = 0; rep <= reps; rep++) {
      auto start = std::chrono::steady_clock::now();
      int num_arg = num;
      int upper_arg = upper;
      int m_arg = m;
      int k_arg = k;
      T alpha_arg = alpha;
      T beta_arg = beta;
      iris::Task task;
      void* params[9] = {&mem_matrices, &mem_vectors, &mem_result, &num_arg, &upper_arg,
                         &m_arg, &k_arg, &alpha_arg, &beta_arg};
      int pinfo[9] = {iris_r, iris_r, iris_w, sizeof(num_arg), sizeof(upper_arg),
                      sizeof(m_arg), sizeof(k_arg), sizeof(alpha_arg), sizeof(beta_arg)};
      task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                  9, params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
      auto end = std::chrono::steady_clock::now();
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3;
      if (rep != 0) {
        sum += elapsed;
      }
      if (verbose) {
        std::cout << "size " << size << ": " << elapsed << " us; "
                  << elapsed / num << " us per operation" << std::endl;
      }
    }

    std::cout << "size " << size << " average execution time: " << sum / reps
              << " us; " << sum / reps / num << " us per operation; "
              << "floating-point operations per second: ";
    performance(m, n, k, 1e3 * (sum / reps / num));

    if constexpr (std::is_same<T, double>::value) {
      mem_result.fetch(result);
      std::memset(result_ref, 0, vectors_size);
      gemmBatched_ref(num, upper, upper, 1, m, k, n, alpha, beta,
                      matrices, lda, vectors, ldb, result_ref, ldc);
      bool mismatch = false;
      for (int i = 0; i < num && !mismatch; i++) {
        for (int j = 0; j < m; j++) {
          if (std::abs(result[i * upper + j] - result_ref[i * upper + j]) > 1e-6) {
            std::cout << "Mismatch at batch index " << i << ": "
                      << result[i * upper + j] << "!=" << result_ref[i * upper + j] << std::endl;
            mismatch = true;
            break;
          }
        }
      }
    }
  }

  std::free(matrices);
  std::free(vectors);
  std::free(result);
  std::free(result_ref);
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int lower = 2;
  int upper = 100;
  int num = 25000;
  int reps = 10;
  int verbose = 0;
  int status;
  while ((status = getopt(argc, argv, "l:u:n:r:v")) != -1) {
    switch (status) {
      case 'l': lower = strtoul(optarg, 0, 0); break;
      case 'u': upper = strtoul(optarg, 0, 0); break;
      case 'n': num = strtoul(optarg, 0, 0); break;
      case 'r': reps = strtoul(optarg, 0, 0); break;
      case 'v': verbose = 1; break;
      default: std::cerr << "invalid argument: " << status << std::endl; return 1;
    }
  }

  std::cout << "running with lower: " << lower << " upper: " << upper
            << " num: " << num << " reps: " << reps << std::endl;
  std::cout << ">>>>>>>>>>>>>>> Half precision gemmBatched >>>>>>>>>>>>>>> " << std::endl;
  gemmBatched<float>(lower, upper, num, reps, verbose, "gemm_batched_float");
  std::cout << ">>>>>>>>>>>>>>> Single precision gemmBatched >>>>>>>>>>>>>>> " << std::endl;
  gemmBatched<float>(lower, upper, num, reps, verbose, "gemm_batched_float");
  std::cout << ">>>>>>>>>>>>>>> Double precision gemmBatched >>>>>>>>>>>>>>> " << std::endl;
  gemmBatched<double>(lower, upper, num, reps, verbose, "gemm_batched_double");

  platform.finalize();
  return 0;
}
