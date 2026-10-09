#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BSIZE 16
#define XTILE 20

using Real = double;

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <grid dimension> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int size = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  const int nx = size;
  const int ny = size;
  const int nz = size;
  const int vol = nx * ny * nz;
  std::printf("Grid dimension: nx=%d ny=%d nz=%d\n", nx, ny, nz);

  std::vector<Real> vm(static_cast<size_t>(vol));
  std::vector<Real> dvm(static_cast<size_t>(vol), 0.0);
  std::vector<Real> sigma_x(static_cast<size_t>(vol) * 3);
  std::vector<Real> sigma_y(static_cast<size_t>(vol) * 3);
  std::vector<Real> sigma_z(static_cast<size_t>(vol) * 3);

#define h_Vm(x, y, z) vm[static_cast<size_t>((z) + nz * ((y) + ny * (x)))]

  for (int ii = 0; ii < nx; ii++) {
    for (int jj = 0; jj < ny; jj++) {
      for (int kk = 0; kk < nz; kk++) {
        h_Vm(ii, jj, kk) = (ii * (ny * nz) + jj * nz + kk) % 19;
      }
    }
  }

  for (int i = 0; i < vol * 3; i++) {
    sigma_x[static_cast<size_t>(i)] = i % 19;
    sigma_y[static_cast<size_t>(i)] = (i + 3 * vol) % 19;
    sigma_z[static_cast<size_t>(i)] = (i + 6 * vol) % 19;
  }

  const int bdimz = (nz - 2) / (BSIZE - 2) + (((nz - 2) % (BSIZE - 2) == 0) ? 0 : 1);
  const int bdimy = (ny - 2) / (BSIZE - 2) + (((ny - 2) % (BSIZE - 2) == 0) ? 0 : 1);
  const int bdimx = (nx - 2) / XTILE + (((nx - 2) % XTILE == 0) ? 0 : 1);

  iris::DMem d_vm(vm.data(), sizeof(Real) * vm.size());
  iris::DMem d_dvm(dvm.data(), sizeof(Real) * dvm.size());
  iris::DMem d_sigma_x(sigma_x.data(), sizeof(Real) * sigma_x.size());
  iris::DMem d_sigma_y(sigma_y.data(), sizeof(Real) * sigma_y.size());
  iris::DMem d_sigma_z(sigma_z.data(), sizeof(Real) * sigma_z.size());

  const size_t gws[3] = {
      static_cast<size_t>(bdimx * BSIZE),
      static_cast<size_t>(bdimy * BSIZE),
      static_cast<size_t>(bdimz)};
  const size_t lws[3] = {BSIZE, BSIZE, 1};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    int nx_arg = nx;
    int ny_arg = ny;
    int nz_arg = nz;
    void* params[8] = {
        &d_vm, &d_dvm, &d_sigma_x, &d_sigma_y, &d_sigma_z,
        &nx_arg, &ny_arg, &nz_arg};
    int pinfo[8] = {
        iris_r, iris_w, iris_r, iris_r, iris_r, sizeof(nx), sizeof(ny), sizeof(nz)};
    task.kernel("stencil3d", 3, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 8, params, pinfo);
    task.flush_out(d_dvm);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  d_dvm.fetch(dvm.data());

#ifdef DUMP
  for (int ii = 0; ii < nx; ii++) {
    for (int jj = 0; jj < ny; jj++) {
      for (int kk = 0; kk < nz; kk++) {
        std::printf("dVm (%d,%d,%d)=%e\n", ii, jj, kk,
                    dvm[static_cast<size_t>(kk + nz * (jj + ny * ii))]);
      }
    }
  }
#endif

  platform.finalize();
  return 0;
}
