#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

#include "gels_types.h"

template <typename T>
static T make_value(double v) { return static_cast<T>(v); }

template <>
ComplexF make_value<ComplexF>(double v) { return ComplexF{0.0f, static_cast<float>(v)}; }

template <>
ComplexD make_value<ComplexD>(double v) { return ComplexD{0.0, v}; }

template <typename T>
static T expected_one() { return static_cast<T>(1.0); }

template <typename T>
static double error_norm(T v)
{
  double d = static_cast<double>(v - expected_one<T>());
  return std::sqrt(d * d);
}

template <>
double error_norm<ComplexF>(ComplexF v)
{
  double dr = static_cast<double>(v.real - 1.0f);
  double di = static_cast<double>(v.imag);
  return std::sqrt(dr * dr + di * di);
}

template <>
double error_norm<ComplexD>(ComplexD v)
{
  double dr = v.real - 1.0;
  double di = v.imag;
  return std::sqrt(dr * dr + di * di);
}

template <typename T>
static void print_value(T v)
{
  std::printf("%6.2f", static_cast<double>(v));
}

template <>
void print_value<ComplexF>(ComplexF v)
{
  std::printf("<%6.2f,%6.2f> ", static_cast<double>(v.real), static_cast<double>(v.imag));
}

template <>
void print_value<ComplexD>(ComplexD v)
{
  std::printf("<%6.2f,%6.2f> ", v.real, v.imag);
}

template <typename T>
static void init_inputs(std::vector<T>& A, std::vector<T>& B)
{
  const double a_vals[] = {
     1.0,  0.0,  0.0,  0.0,  0.0,
     1.0,  0.2, -0.4, -0.4, -0.8,
     1.0,  0.6, -0.2,  0.4, -1.2,
     1.0,  1.0, -1.0,  0.6, -0.8,
     1.0,  1.8, -0.6,  0.2, -0.6,
     0.2, -0.4, -0.4, -0.8,  0.0,
     0.4,  0.2,  0.8, -0.4,  0.0,
     0.4, -0.8,  0.2,  0.4,  0.0,
     0.8,  0.4, -0.4,  0.2,  0.0,
     0.0,  0.0,  0.0,  0.0,  1.0
  };
  const double b_vals[] = {
     5.0,  3.6, -2.2,  0.8, -3.4,
     1.8, -0.6,  0.2, -0.6,  1.0
  };
  for (int i = 0; i < 50; i++) A[i] = make_value<T>(a_vals[i]);
  for (int i = 0; i < 10; i++) B[i] = make_value<T>(b_vals[i]);
}

template <typename T>
static int run_gels_batch_example(const char* kernel_name, const int repeat, iris::Platform& platform)
{
  (void)platform;
  constexpr int n = 5;
  constexpr int stride_a = 25;
  constexpr int stride_b = 5;
  constexpr int batch_size = 2;

  std::vector<T> A(stride_a * batch_size);
  std::vector<T> B(stride_b * batch_size);
  init_inputs(A, B);

  iris::DMem mem_A(A.data(), A.size() * sizeof(T));
  iris::DMem mem_B(B.data(), B.size() * sizeof(T));

  const size_t gws[1] = { batch_size };
  const size_t lws[1] = { 1 };
  long time = 0;
  for (int i = 0; i <= repeat; i++) {
    init_inputs(A, B);
    mem_A.update(A.data());
    mem_B.update(B.data());
    iris::Task task;
    void* params[2] = { &mem_A, &mem_B };
    int pinfo[2] = { iris_r, iris_rw };
    auto start = std::chrono::steady_clock::now();
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 2, params, pinfo);
    task.flush_out(mem_B);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();
    if (i != 0) time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  std::printf("Average kernel execution time : %f (us)\n", (time * 1e-3f) / repeat);

  const double bound = std::is_same_v<T, float> || std::is_same_v<T, ComplexF> ? 1e-6 : 1e-8;
  bool passed = true;
  std::printf("Results:\n");
  for (int batch = 0; batch < batch_size; batch++) {
    for (int j = 0; j < n; j++) {
      T result = B[batch * stride_b + j];
      passed = passed && error_norm(result) < bound;
      print_value(result);
    }
    std::printf("\n");
  }

  if (passed) {
    std::printf("Calculations successfully finished\n");
    return 0;
  }
  std::printf("ERROR: results mismatch!\n");
  return 1;
}

static void print_info()
{
  std::cout << "\n########################################################################\n";
  std::cout << "# Batched strided GELS example:\n";
  std::cout << "# \n";
  std::cout << "# Computes least squares of a batch of matrices and right hand sides.\n";
  std::cout << "# Supported floating point type precisions:\n";
  std::cout << "#   float\n";
  std::cout << "#   double\n";
  std::cout << "#   std::complex<float>\n";
  std::cout << "#   std::complex<double>\n";
  std::cout << "# \n";
  std::cout << "########################################################################\n\n";
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  print_info();

  bool failed = false;
  std::cout << "Running with single precision real data type:" << std::endl;
  failed |= run_gels_batch_example<float>("gels_float", repeat, platform);
  std::cout << "Running with single precision complex data type:" << std::endl;
  failed |= run_gels_batch_example<ComplexF>("gels_cfloat", repeat, platform);
  std::cout << "Running with double precision real data type:" << std::endl;
  failed |= run_gels_batch_example<double>("gels_double", repeat, platform);
  std::cout << "Running with double precision complex data type:" << std::endl;
  failed |= run_gels_batch_example<ComplexD>("gels_cdouble", repeat, platform);

  platform.finalize();
  return failed ? 1 : 0;
}
