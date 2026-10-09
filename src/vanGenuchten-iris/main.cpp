#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "reference.h"

static void launch_van_genuchten(
    iris::DMem& mem_Ksat,
    iris::DMem& mem_psi,
    iris::DMem& mem_C,
    iris::DMem& mem_theta,
    iris::DMem& mem_K,
    int size)
{
  const size_t block = 256;
  size_t gws[1] = { ((static_cast<size_t>(size) + block - 1) / block) * block };
  size_t lws[1] = { block };

  iris::Task task;
  void* params[6] = {
    &mem_Ksat,
    &mem_psi,
    &mem_C,
    &mem_theta,
    &mem_K,
    &size
  };
  int pinfo[6] = {
    iris_r,
    iris_r,
    iris_w,
    iris_w,
    iris_w,
    sizeof(size)
  };

  task.kernel("vanGenuchten", 1, nullptr, gws, lws, 6, params, pinfo);
  task.flush_out(mem_C);
  task.flush_out(mem_theta);
  task.flush_out(mem_K);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: ./%s <dimX> <dimY> <dimZ> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int dimX = std::atoi(argv[1]);
  const int dimY = std::atoi(argv[2]);
  const int dimZ = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  const int size = dimX * dimY * dimZ;
  const int size_byte = size * sizeof(double);

  double* Ksat = new double[size];
  double* psi = new double[size];
  double* C = new double[size];
  double* theta = new double[size];
  double* K = new double[size];
  double* C_ref = new double[size];
  double* theta_ref = new double[size];
  double* K_ref = new double[size];

  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 + (1.0 - 1e-6) * i / size;
    psi[i] = -100.0 + 101.0 * i / size;
  }

  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  iris::DMem mem_Ksat(Ksat, size_byte);
  iris::DMem mem_psi(psi, size_byte);
  iris::DMem mem_C(C, size_byte);
  iris::DMem mem_theta(theta, size_byte);
  iris::DMem mem_K(K, size_byte);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_van_genuchten(mem_Ksat, mem_psi, mem_C, mem_theta, mem_K, size);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  mem_C.fetch(C);
  mem_theta.fetch(theta);
  mem_K.fetch(K);

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (std::fabs(C[i] - C_ref[i]) > 1e-3 ||
        std::fabs(theta[i] - theta_ref[i]) > 1e-3 ||
        std::fabs(K[i] - K_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  delete[] Ksat;
  delete[] psi;
  delete[] C;
  delete[] theta;
  delete[] K;
  delete[] C_ref;
  delete[] theta_ref;
  delete[] K_ref;

  platform.finalize();
  return 0;
}
