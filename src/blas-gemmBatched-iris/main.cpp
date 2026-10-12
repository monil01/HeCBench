#include <iris/iris.hpp>

#include <hip/hip_fp16.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "reference.h"

using namespace std;

constexpr int kThreads = 256;

template <typename T>
const char* kernel_name();

template <>
const char* kernel_name<float>() { return "gemm_batched_float"; }

template <>
const char* kernel_name<double>() { return "gemm_batched_double"; }

template <>
const char* kernel_name<__half>() { return "gemm_batched_half"; }

template <typename T>
T random_value()
{
  return static_cast<T>(drand48());
}

template <>
__half random_value<__half>()
{
  return __float2half(static_cast<float>(drand48()));
}

template <typename T>
void gemmBatched(int lower, int upper, int num, int reps, int verbose)
{
  if (verbose) cout << "initializing inputs" << endl;
  size_t matrices_size = static_cast<size_t>(upper) * upper * num * sizeof(T);
  size_t vectors_size = static_cast<size_t>(upper) * num * sizeof(T);

  T* matrices = static_cast<T*>(malloc(matrices_size));
  T* vectors = static_cast<T*>(malloc(vectors_size));
  T* result = static_cast<T*>(malloc(vectors_size));
  T* result_ref = static_cast<T*>(malloc(vectors_size));
  assert(matrices && vectors && result && result_ref);

  srand48(48);
  for (int i = 0; i < num * upper * upper; i++) matrices[i] = random_value<T>();
  for (int i = 0; i < num * upper; i++) vectors[i] = random_value<T>();
  memset(result, 0, vectors_size);
  memset(result_ref, 0, vectors_size);

  iris::DMem mem_matrices(matrices, matrices_size);
  iris::DMem mem_vectors(vectors, vectors_size);
  iris::DMem mem_result(result, vectors_size);

  for (int size = lower; size <= upper; size++) {
    if (verbose) cout << "running with <size x size> x <size x 1> " << size << endl;
    double sum = 0.0;
    const int m = size, n = 1, k = size;
    for (int rep = 0; rep <= reps; rep++) {
      memset(result, 0, vectors_size);
      mem_result.update(result);

      size_t gws[1] = {
          static_cast<size_t>(((num * size) + kThreads - 1) / kThreads) * kThreads};
      size_t lws[1] = {kThreads};
      iris::Task task;
      void* params[6] = {&mem_matrices, &mem_vectors, &mem_result, &upper, &size, &num};
      int pinfo[6] = {iris_r, iris_r, iris_w, sizeof(upper), sizeof(size), sizeof(num)};

      auto start = chrono::steady_clock::now();
      task.kernel(kernel_name<T>(), 1, nullptr, gws, lws, 6, params, pinfo);
      task.flush_out(mem_result);
      task.submit(iris_gpu, nullptr, 1);
      auto end = chrono::steady_clock::now();

      auto time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
      auto elapsed = time * 1e-3;
      if (rep != 0) sum += elapsed;

      if (verbose) {
        cout << "size " << size << ": " << elapsed << " us; "
             << elapsed / num << " us per operation" << endl;
      }
    }

    cout << "size " << size << " average execution time: " << sum / reps << " us; "
         << sum / reps / num << " us per operation; "
         << "floating-point operations per second: ";
    performance(m, n, k, 1e3 * (sum / reps / num));

    if constexpr (is_same_v<T, double>) {
      gemmBatched_ref(num, upper, upper, 1, m, k, n, 1.0, 0.0, matrices, upper,
                      vectors, upper, result_ref, upper);
      for (int i = 0; i < num; i++) {
        for (int j = 0; j < m; j++) {
          if (abs(result[i * upper + j] - result_ref[i * upper + j]) > 1e-6) {
            cout << "Mismatch at batch index " << i << ": "
                 << result[i * upper + j] << "!=" << result_ref[i * upper + j]
                 << endl;
            break;
          }
        }
      }
    }
  }

  free(matrices);
  free(vectors);
  free(result);
  free(result_ref);
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int status;
  int lower = 2;
  int upper = 100;
  int num = 25000;
  int reps = 10;
  int verbose = 0;

  while ((status = getopt(argc, argv, "l:u:n:r:v")) != -1) {
    switch (status) {
    case 'l':
      lower = strtoul(optarg, 0, 0);
      break;
    case 'u':
      upper = strtoul(optarg, 0, 0);
      break;
    case 'n':
      num = strtoul(optarg, 0, 0);
      break;
    case 'r':
      reps = strtoul(optarg, 0, 0);
      break;
    case 'v':
      verbose = 1;
      break;
    default:
      cerr << "invalid argument: " << status << endl;
      platform.finalize();
      return 1;
    }
  }

  cout << "running with lower: " << lower << " upper: " << upper
       << " num: " << num << " reps: " << reps << endl;
  cout << ">>>>>>>>>>>>>>> Half precision gemmBatched >>>>>>>>>>>>>>> " << endl;
  gemmBatched<__half>(lower, upper, num, reps, verbose);
  cout << ">>>>>>>>>>>>>>> Single precision gemmBatched >>>>>>>>>>>>>>> " << endl;
  gemmBatched<float>(lower, upper, num, reps, verbose);
  cout << ">>>>>>>>>>>>>>> Double precision gemmBatched >>>>>>>>>>>>>>> " << endl;
  gemmBatched<double>(lower, upper, num, reps, verbose);

  platform.finalize();
  return 0;
}
