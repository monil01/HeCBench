#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400

typedef double nRarray[DATAYSIZE][DATAZSIZE];

#define SQ(x) ((x) * (x))

#include "../ace-cuda/reference.h"

static void initializationPhi(double phi[][DATAYSIZE][DATAZSIZE], double r0) {
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx - 0.5 * DATAXSIZE) +
                             SQ(idy - 0.5 * DATAYSIZE) +
                             SQ(idz - 0.5 * DATAZSIZE));
        phi[idx][idy][idz] = (r < r0) ? 1.0 : -1.0;
      }
    }
  }
}

static void initializationU(double u[][DATAYSIZE][DATAZSIZE], double r0, double delta) {
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx - 0.5 * DATAXSIZE) +
                             SQ(idy - 0.5 * DATAYSIZE) +
                             SQ(idz - 0.5 * DATAZSIZE));
        u[idx][idy][idz] = (r < r0) ? 0.0 : -delta * (1.0 - std::exp(-(r - r0)));
      }
    }
  }
}

static void run_kernel(const char* name,
                       iris::DMem& a,
                       iris::DMem& b,
                       iris::DMem& c,
                       iris::DMem& d,
                       iris::DMem& e,
                       iris::DMem& f,
                       iris::DMem& g,
                       const double* scalars,
                       int scalar_count,
                       const int* pinfo,
                       void** params,
                       int param_count,
                       size_t* gws,
                       size_t* lws) {
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  (void)e;
  (void)f;
  (void)g;
  (void)scalars;
  (void)scalar_count;
  iris::Task task;
  task.kernel(name, 3, nullptr, gws, lws, param_count, params, const_cast<int*>(pinfo));
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[]) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  if (argc < 2) {
    std::printf("Usage: ./main num_steps\n");
    platform.finalize();
    return 255;
  }

  const int num_steps = std::atoi(argv[1]);
  double dx = 0.4;
  double dy = 0.4;
  double dz = 0.4;
  double dt = 0.01;
  double delta = 0.8;
  double r0 = 5.0;
  double epsilon = 0.07;
  double W0 = 1.0;
  double beta0 = 0.0;
  double D = 2.0;
  double d0 = 0.5;
  double a1 = 1.25 / std::sqrt(2.0);
  double a2 = 0.64;
  double lambda = (W0 * a1) / d0;
  double tau0 = ((W0 * W0 * W0 * a1 * a2) / (d0 * D)) + ((W0 * W0 * beta0) / d0);

  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;
  const size_t vol_in_bytes = sizeof(double) * static_cast<size_t>(vol);

  nRarray* phi_host = static_cast<nRarray*>(std::malloc(vol_in_bytes));
  nRarray* u_host = static_cast<nRarray*>(std::malloc(vol_in_bytes));
  nRarray* phinew_host = static_cast<nRarray*>(std::malloc(vol_in_bytes));
  nRarray* unew_host = static_cast<nRarray*>(std::malloc(vol_in_bytes));
  nRarray* Fx_host = static_cast<nRarray*>(std::malloc(vol_in_bytes));
  nRarray* Fy_host = static_cast<nRarray*>(std::malloc(vol_in_bytes));
  nRarray* Fz_host = static_cast<nRarray*>(std::malloc(vol_in_bytes));

  initializationPhi(phi_host, r0);
  initializationU(u_host, r0, delta);
  std::memset(phinew_host, 0, vol_in_bytes);
  std::memset(unew_host, 0, vol_in_bytes);
  std::memset(Fx_host, 0, vol_in_bytes);
  std::memset(Fy_host, 0, vol_in_bytes);
  std::memset(Fz_host, 0, vol_in_bytes);

#ifdef VERIFY
  nRarray* phi_ref = static_cast<nRarray*>(std::malloc(vol_in_bytes));
  nRarray* u_ref = static_cast<nRarray*>(std::malloc(vol_in_bytes));
  std::memcpy(phi_ref, phi_host, vol_in_bytes);
  std::memcpy(u_ref, u_host, vol_in_bytes);
  reference(phi_ref, u_ref, vol, num_steps);
#endif

  iris::DMem mem_phiold(phi_host, vol_in_bytes);
  iris::DMem mem_phinew(phinew_host, vol_in_bytes);
  iris::DMem mem_uold(u_host, vol_in_bytes);
  iris::DMem mem_unew(unew_host, vol_in_bytes);
  iris::DMem mem_Fx(Fx_host, vol_in_bytes);
  iris::DMem mem_Fy(Fy_host, vol_in_bytes);
  iris::DMem mem_Fz(Fz_host, vol_in_bytes);

  size_t gws[3] = {((DATAZSIZE + 7) / 8) * 8u,
                   ((DATAYSIZE + 7) / 8) * 8u,
                   ((DATAXSIZE + 3) / 4) * 4u};
  size_t lws[3] = {8, 8, 4};

  auto start = std::chrono::steady_clock::now();

  for (int t = 0; t <= num_steps; ++t) {
    {
      void* params[] = {&mem_phiold, &mem_Fx, &mem_Fy, &mem_Fz,
                        &dx, &dy, &dz, &epsilon, &W0, &tau0};
      int pinfo[] = {iris_r, iris_w, iris_w, iris_w,
                     sizeof(dx), sizeof(dy), sizeof(dz),
                     sizeof(epsilon), sizeof(W0), sizeof(tau0)};
      run_kernel("calculateForce", mem_phiold, mem_Fx, mem_Fy, mem_Fz,
                 mem_uold, mem_phinew, mem_unew, nullptr, 0,
                 pinfo, params, 10, gws, lws);
    }
    {
      void* params[] = {&mem_phinew, &mem_phiold, &mem_uold,
                        &mem_Fx, &mem_Fy, &mem_Fz,
                        &epsilon, &W0, &tau0, &lambda,
                        &dt, &dx, &dy, &dz};
      int pinfo[] = {iris_w, iris_r, iris_r, iris_r, iris_r, iris_r,
                     sizeof(epsilon), sizeof(W0), sizeof(tau0), sizeof(lambda),
                     sizeof(dt), sizeof(dx), sizeof(dy), sizeof(dz)};
      run_kernel("allenCahn", mem_phinew, mem_phiold, mem_uold, mem_Fx,
                 mem_Fy, mem_Fz, mem_unew, nullptr, 0,
                 pinfo, params, 14, gws, lws);
    }
    {
      void* params[] = {&mem_phinew};
      int pinfo[] = {iris_rw};
      run_kernel("boundaryConditionsPhi", mem_phinew, mem_phiold, mem_uold,
                 mem_Fx, mem_Fy, mem_Fz, mem_unew, nullptr, 0,
                 pinfo, params, 1, gws, lws);
    }
    {
      void* params[] = {&mem_unew, &mem_uold, &mem_phinew, &mem_phiold,
                        &D, &dt, &dx, &dy, &dz};
      int pinfo[] = {iris_w, iris_r, iris_r, iris_r,
                     sizeof(D), sizeof(dt), sizeof(dx), sizeof(dy), sizeof(dz)};
      run_kernel("thermalEquation", mem_unew, mem_uold, mem_phinew, mem_phiold,
                 mem_Fx, mem_Fy, mem_Fz, nullptr, 0,
                 pinfo, params, 9, gws, lws);
    }
    {
      void* params[] = {&mem_unew, &delta};
      int pinfo[] = {iris_rw, sizeof(delta)};
      run_kernel("boundaryConditionsU", mem_unew, mem_uold, mem_phinew,
                 mem_phiold, mem_Fx, mem_Fy, mem_Fz, nullptr, 0,
                 pinfo, params, 2, gws, lws);
    }
    {
      void* params[] = {&mem_phinew, &mem_phiold};
      int pinfo[] = {iris_rw, iris_rw};
      run_kernel("swapGrid", mem_phinew, mem_phiold, mem_uold, mem_unew,
                 mem_Fx, mem_Fy, mem_Fz, nullptr, 0,
                 pinfo, params, 2, gws, lws);
    }
    {
      void* params[] = {&mem_unew, &mem_uold};
      int pinfo[] = {iris_rw, iris_rw};
      iris::Task task;
      task.kernel("swapGrid", 3, nullptr, gws, lws, 2, params, pinfo);
      if (t == num_steps) {
        task.flush_out(mem_phiold);
        task.flush_out(mem_uold);
      }
      task.submit(iris_gpu, nullptr, 1);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Total kernel execution time: %.3f (ms)\n", time * 1e-6f);
  std::printf("Offload time: %.3f (ms)\n", time * 1e-6f);

#ifdef VERIFY
  bool ok = true;
  for (int idx = 0; idx < nx; idx++) {
    for (int idy = 0; idy < ny; idy++) {
      for (int idz = 0; idz < nz; idz++) {
        if (std::fabs(phi_ref[idx][idy][idz] - phi_host[idx][idy][idz]) > 1e-3) {
          ok = false;
          std::printf("phi: %lf %lf\n", phi_ref[idx][idy][idz], phi_host[idx][idy][idz]);
        }
        if (std::fabs(u_ref[idx][idy][idz] - u_host[idx][idy][idz]) > 1e-3) {
          ok = false;
          std::printf("u: %lf %lf\n", u_ref[idx][idy][idz], u_host[idx][idy][idz]);
        }
      }
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  std::free(phi_ref);
  std::free(u_ref);
#endif

  std::free(phi_host);
  std::free(u_host);
  std::free(phinew_host);
  std::free(unew_host);
  std::free(Fx_host);
  std::free(Fy_host);
  std::free(Fz_host);

  platform.finalize();
  return 0;
}
