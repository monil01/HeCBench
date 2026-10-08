#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "che.h"

static void initialization(double* c)
{
  std::srand(2);
  for (int idz = 0; idz < DATAZSIZE; idz++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idx = 0; idx < DATAXSIZE; idx++) {
        double f = static_cast<double>(std::rand()) / RAND_MAX;
        c[(idz * DATAYSIZE + idy) * DATAXSIZE + idx] = -1.0 + 2.0 * f;
      }
    }
  }
}

static double integral(const double* c, int nx, int ny, int nz)
{
  double summation = 0.0;
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        summation += c[(k * ny + j) * nx + i];
      }
    }
  }
  return summation;
}

static void submit_field_kernel(const char* name,
                                iris::DMem& in,
                                iris::DMem& out,
                                double dx,
                                double dy,
                                double dz,
                                double gamma,
                                double e_AA,
                                double e_BB,
                                double e_AB)
{
  size_t gws[3] = {DATAXSIZE, DATAYSIZE, DATAZSIZE};
  size_t lws[3] = {BLKXSIZE, BLKYSIZE, BLKZSIZE};
  iris::Task task;
  void* params[9] = {&in, &out, &dx, &dy, &dz, &gamma, &e_AA, &e_BB, &e_AB};
  int pinfo[9] = {iris_r, iris_w, sizeof(dx), sizeof(dy), sizeof(dz),
                  sizeof(gamma), sizeof(e_AA), sizeof(e_BB), sizeof(e_AB)};
  task.kernel(name, 3, nullptr, gws, lws, 9, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void submit_cahn_hilliard(iris::DMem& cnew,
                                 iris::DMem& cold,
                                 iris::DMem& mu,
                                 double D,
                                 double dt,
                                 double dx,
                                 double dy,
                                 double dz)
{
  size_t gws[3] = {DATAXSIZE, DATAYSIZE, DATAZSIZE};
  size_t lws[3] = {BLKXSIZE, BLKYSIZE, BLKZSIZE};
  iris::Task task;
  void* params[8] = {&cnew, &cold, &mu, &D, &dt, &dx, &dy, &dz};
  int pinfo[8] = {iris_w, iris_r, iris_r, sizeof(D), sizeof(dt),
                  sizeof(dx), sizeof(dy), sizeof(dz)};
  task.kernel("cahnHilliard", 3, nullptr, gws, lws, 8, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void submit_swap(iris::DMem& cnew, iris::DMem& cold)
{
  size_t gws[3] = {DATAXSIZE, DATAYSIZE, DATAZSIZE};
  size_t lws[3] = {BLKXSIZE, BLKYSIZE, BLKZSIZE};
  iris::Task task;
  void* params[2] = {&cnew, &cold};
  int pinfo[2] = {iris_rw, iris_rw};
  task.kernel("Swap", 3, nullptr, gws, lws, 2, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void fetch_to_host(iris::DMem& mem, double* host)
{
  iris::Task task;
  task.flush_out(mem);
  task.submit(iris_gpu, nullptr, 1);
  mem.fetch(host);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  double dx = 1.0;
  double dy = 1.0;
  double dz = 1.0;
  double dt = 0.01;
  double e_AA = -(2.0 / 9.0);
  double e_BB = -(2.0 / 9.0);
  double e_AB = (2.0 / 9.0);
  int t_f = std::atoi(argv[1]);
#ifndef DEBUG
  int t_freq = t_f;
#else
  int t_freq = 10;
#endif
  double gamma = 0.5;
  double D = 1.0;

  std::ofstream ofile_c("./out/integral_c.txt");
  std::ofstream ofile_mu("./out/integral_mu.txt");
  std::ofstream ofile_f("./out/integral_f.txt");

  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;
  const size_t vol_bytes = static_cast<size_t>(vol) * sizeof(double);

  double* c_host = static_cast<double*>(std::malloc(vol_bytes));
  double* mu_host = static_cast<double*>(std::malloc(vol_bytes));
  double* f_host = static_cast<double*>(std::malloc(vol_bytes));
  double* cnew_host = static_cast<double*>(std::malloc(vol_bytes));
  if (!c_host || !mu_host || !f_host || !cnew_host) {
    std::fprintf(stderr, "host malloc failed\n");
    std::free(c_host);
    std::free(mu_host);
    std::free(f_host);
    std::free(cnew_host);
    platform.finalize();
    return 1;
  }

  initialization(c_host);

  iris::DMem d_cold(c_host, vol_bytes);
  iris::DMem d_cnew(cnew_host, vol_bytes);
  iris::DMem d_muold(mu_host, vol_bytes);
  iris::DMem d_fold(f_host, vol_bytes);

  double integral_c = 0.0;
  double integral_mu = 0.0;
  double integral_f = 0.0;

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < t_f; t++) {
    submit_field_kernel("chemicalPotential", d_cold, d_muold,
                        dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    submit_field_kernel("localFreeEnergyFunctional", d_cold, d_fold,
                        dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    submit_cahn_hilliard(d_cnew, d_cold, d_muold, D, dt, dx, dy, dz);

    if (t > 0 && t % (t_freq - 1) == 0) {
      fetch_to_host(d_cnew, c_host);
      fetch_to_host(d_muold, mu_host);
      fetch_to_host(d_fold, f_host);

      integral_c = integral(c_host, nx, ny, nz);
      ofile_c << t << "," << integral_c << "\n";

      integral_mu = integral(mu_host, nx, ny, nz);
      ofile_mu << t << "," << integral_mu << "\n";

      integral_f = integral(f_host, nx, ny, nz);
      ofile_f << t << "," << integral_f << "\n";
    }

    submit_swap(d_cnew, d_cold);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Kernel exeuction time on the GPU (%d iterations) = %.3f (s)\n",
              t_f, time * 1e-9f);

  std::free(c_host);
  std::free(mu_host);
  std::free(f_host);
  std::free(cnew_host);
  platform.finalize();
  return 0;
}
