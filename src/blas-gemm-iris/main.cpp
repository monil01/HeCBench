#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "utils.h"

template <typename fp>
static void gemm_cpu(const fp* a, const fp* b, const fp* c_initial, fp* ref,
                     int m, int k, int n, fp alpha, fp beta)
{
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      fp sum = fp(0);
      for (int kk = 0; kk < k; kk++) {
        sum += a[row * k + kk] * b[kk * n + col];
      }
      ref[row * n + col] = alpha * sum + beta * c_initial[row * n + col];
    }
  }
}

template <typename fp>
static int run_gemm_example(int m, int k, int n, int repeat, const char* kernel_name)
{
  const fp alpha = fp(2.0);
  const fp beta = fp(0.5);
  const size_t A_size = sizeof(fp) * static_cast<size_t>(m) * k;
  const size_t B_size = sizeof(fp) * static_cast<size_t>(k) * n;
  const size_t C_size = sizeof(fp) * static_cast<size_t>(m) * n;

  fp* a = static_cast<fp*>(aligned_alloc(64, A_size));
  fp* b = static_cast<fp*>(aligned_alloc(64, B_size));
  fp* c = static_cast<fp*>(aligned_alloc(64, C_size));
  fp* c_initial = static_cast<fp*>(aligned_alloc(64, C_size));
  fp* ref = static_cast<fp*>(aligned_alloc(64, C_size));
  if (!a || !b || !c || !c_initial || !ref) {
    std::cerr << "allocation failed" << std::endl;
    return 1;
  }

  std::srand(2);
  rand_matrix(a, m, k);
  rand_matrix(b, k, n);
  rand_matrix(c, m, n);
  std::memcpy(c_initial, c, C_size);

  iris::DMem mem_a(a, A_size);
  iris::DMem mem_b(b, B_size);
  iris::DMem mem_c(c, C_size);

  std::cout << "Checking BLAS GEMM.. ";
  const size_t lws[2] = {16, 16};
  const size_t gws[2] = {
      static_cast<size_t>((n + 15) / 16 * 16),
      static_cast<size_t>((m + 15) / 16 * 16)};

  auto launch = [&]() {
    int m_arg = m;
    int k_arg = k;
    int n_arg = n;
    fp alpha_arg = alpha;
    fp beta_arg = beta;
    iris::Task task;
    void* params[8] = {&mem_a, &mem_b, &mem_c, &m_arg, &k_arg, &n_arg, &alpha_arg, &beta_arg};
    int pinfo[8] = {iris_r, iris_r, iris_rw, sizeof(m_arg), sizeof(k_arg), sizeof(n_arg),
                    sizeof(alpha_arg), sizeof(beta_arg)};
    task.kernel(kernel_name, 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                8, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  };

  launch();
  mem_c.fetch(c);
  gemm_cpu(a, b, c_initial, ref, m, k, n, alpha, beta);

  bool ok = true;
  const double tol = std::is_same<fp, double>::value ? 1e-9 : 1e-4;
  for (int i = 0; i < m * n; i++) {
    if (std::fabs(static_cast<double>(c[i] - ref[i])) > tol) {
      ok = false;
      break;
    }
  }
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  std::memcpy(c, c_initial, C_size);
  mem_c.update(c);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch();
  }
  iris::Task flush;
  flush.flush_out(mem_c);
  flush.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  const double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  performance(m, n, k, false, time / repeat);

  free(a);
  free(b);
  free(c);
  free(c_initial);
  free(ref);
  return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    std::printf("Usage: %s <m> <k> <n> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int m = std::atoi(argv[1]);
  const int k = std::atoi(argv[2]);
  const int n = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  std::cout << "\tRunning with half precision data type:" << std::endl;
  int status = run_gemm_example<float>(m, k, n, repeat, "matrix_mul_float");
  std::cout << "\tRunning with single precision data type:" << std::endl;
  status |= run_gemm_example<float>(m, k, n, repeat, "matrix_mul_float");
  std::cout << "\tRunning with double precision data type:" << std::endl;
  status |= run_gemm_example<double>(m, k, n, repeat, "matrix_mul_double");

  platform.finalize();
  return status;
}
