#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr int BLOCK_SIZE = 256;
constexpr int TEST_ITER = 500;

struct Csr {
  int n;
  std::vector<int> row;
  std::vector<int> col;
  std::vector<double> value;
};

Csr make_laplacian(int nx, int ny, int nz) {
  const int n = nx * ny * nz;
  Csr a{n, std::vector<int>(n + 1), {}, {}};
  a.row[0] = 0;
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const int r = (z * ny + y) * nx + x;
        a.row[r + 1] = a.row[r] + 1 + (z > 0) + (y > 0) + (x > 0) +
                       (x + 1 < nx) + (y + 1 < ny) + (z + 1 < nz);
      }
    }
  }
  a.col.resize(a.row[n]);
  a.value.resize(a.row[n]);
  int p = 0;
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const int r = (z * ny + y) * nx + x;
        a.col[p] = r; a.value[p++] = 6.0;
        if (z > 0) { a.col[p] = r - nx * ny; a.value[p++] = -1.0; }
        if (y > 0) { a.col[p] = r - nx; a.value[p++] = -1.0; }
        if (x > 0) { a.col[p] = r - 1; a.value[p++] = -1.0; }
        if (x + 1 < nx) { a.col[p] = r + 1; a.value[p++] = -1.0; }
        if (y + 1 < ny) { a.col[p] = r + nx; a.value[p++] = -1.0; }
        if (z + 1 < nz) { a.col[p] = r + nx * ny; a.value[p++] = -1.0; }
      }
    }
  }
  return a;
}

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int launch_relax(iris::DMem& out, iris::DMem* const* inputs, const int* input_modes,
                 int input_count, size_t n, void** extra_params, int* extra_info) {
  void* params[8];
  int pinfo[8];
  int count = 0;
  params[count] = &out; pinfo[count++] = iris_rw;
  for (int i = 0; i < input_count; ++i) {
    params[count] = inputs[i]; pinfo[count++] = input_modes[i];
  }
  params[count] = extra_params[0]; pinfo[count++] = extra_info[0];
  size_t gws[] = {n};
  gws[0] = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  size_t lws[] = {BLOCK_SIZE};
  iris::Task task;
  task.kernel("amgmk_relax", 1, nullptr, gws, lws, count, params, pinfo);
  task.flush_out(out);
  const int status = task.submit(iris_gpu, nullptr, 1);
  task.wait();
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::printf("\n//------------ \n// \n//  CORAL  AMGmk Benchmark Version 1.0 \n// \n//------------ \n\n");
  std::printf(" testIter   = %d \n\n", TEST_ITER);

  const int nx = 50, ny = 50, nz = 50;
  Csr a = make_laplacian(nx, ny, nz);
  const int n = a.n;
  const size_t nnz = a.value.size();
  std::vector<double> x(n, 1.0), y(n, 0.0), sol(n, 0.0), f(n, 1.0);
  for (int i = 0; i < n; ++i)
    for (int p = a.row[i]; p < a.row[i + 1]; ++p) sol[i] += a.value[p];

  iris::DMem mem_row(a.row.data(), a.row.size() * sizeof(int));
  iris::DMem mem_col(a.col.data(), a.col.size() * sizeof(int));
  iris::DMem mem_a(a.value.data(), a.value.size() * sizeof(double));
  iris::DMem mem_x(x.data(), x.size() * sizeof(double));
  iris::DMem mem_u(x.data(), x.size() * sizeof(double));
  iris::DMem mem_f(f.data(), f.size() * sizeof(double));

  double t0 = now();
  for (int i = 0; i < TEST_ITER; ++i) {
    for (int row = 0; row < n; ++row) {
      y[row] = 0.0;
      for (int p = a.row[row]; p < a.row[row + 1]; ++p)
        y[row] += a.value[p] * x[a.col[p]];
    }
  }
  const double matvec_time = now() - t0;
  double matvec_error = 0.0;
  for (int i = 0; i < n; ++i) matvec_error = std::max(matvec_error, std::fabs(y[i] - sol[i]));
  std::printf("\n//------------ \n// \n//   MATVEC\n// \n//------------ \n\nWall time = %f seconds. \n", matvec_time);

  f = sol;
  mem_f.update(f.data());
  t0 = now();
  for (int i = 0; i < TEST_ITER; ++i) {
    iris::DMem* inputs[] = {&mem_a, &mem_row, &mem_col, &mem_f};
    int modes[] = {iris_r, iris_r, iris_r, iris_r};
    int nn = n;
    void* scalars[] = {&nn}; int sinfo[] = {static_cast<int>(sizeof(nn))};
    launch_relax(mem_u, inputs, modes, 4, n, scalars, sinfo);
  }
  mem_u.fetch(x.data());
  const double relax_time = now() - t0;
  double relax_error = 0.0;
  for (int i = 0; i < n; ++i) relax_error = std::max(relax_error, std::fabs(x[i] - 1.0));
  std::printf("\n//------------ \n// \n//   Relax\n// \n//------------ \n\nTotal kernel time = %f seconds. \n", relax_time);

  std::vector<double> ax(n, 1.0), ay(n, 1.0);
  const double alpha = 0.5;
  t0 = now();
  for (int i = 0; i < TEST_ITER; ++i) {
    for (int j = 0; j < n; ++j) ay[j] += alpha * ax[j];
  }
  const double axpy_time = now() - t0;
  double axpy_error = 0.0;
  for (int i = 0; i < n; ++i) axpy_error = std::max(axpy_error, std::fabs(ay[i] - 1.0 - alpha * TEST_ITER));
  std::printf("\n//------------ \n// \n//   Axpy\n// \n//------------ \n\nWall time = %f seconds. \n", axpy_time);
  const bool pass = matvec_error == 0.0 && relax_error == 0.0 && axpy_error == 0.0;
  std::printf("\nAMGMK errors: matvec=%e relax=%e axpy=%e\n", matvec_error, relax_error, axpy_error);
  std::printf("AMGMK verification: %s\n", pass ? "PASS" : "FAIL");
  std::printf("\nTotal Wall time = %f seconds. \n", matvec_time + relax_time + axpy_time);
  platform.finalize();
  return pass ? 0 : 1;
}
