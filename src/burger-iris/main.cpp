#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define idx(i, j) ((i) * x_points + (j))

static void initialize(double* x, double* y, double* u, double* v,
                       double* u_new, double* v_new, int x_points,
                       int y_points, double del_x, double del_y)
{
  for (int i = 0; i < x_points; i++) x[i] = i * del_x;
  for (int i = 0; i < y_points; i++) y[i] = i * del_y;
  for (int i = 0; i < y_points; i++) {
    for (int j = 0; j < x_points; j++) {
      u[idx(i, j)] = 1.0;
      v[idx(i, j)] = 1.0;
      u_new[idx(i, j)] = 1.0;
      v_new[idx(i, j)] = 1.0;
      if (x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0) {
        u[idx(i, j)] = 2.0;
        v[idx(i, j)] = 2.0;
        u_new[idx(i, j)] = 2.0;
        v_new[idx(i, j)] = 2.0;
      }
    }
  }
}

static void serial_reference(double* u, double* v, double* u_new, double* v_new,
                             int x_points, int y_points, int num_itrs,
                             double nu, double del_t, double del_x, double del_y)
{
  for (int itr = 0; itr < num_itrs; itr++) {
    for (int i = 1; i < y_points - 1; i++) {
      for (int j = 1; j < x_points - 1; j++) {
        u_new[idx(i, j)] = u[idx(i, j)] +
          (nu * del_t / (del_x * del_x)) * (u[idx(i, j + 1)] + u[idx(i, j - 1)] - 2 * u[idx(i, j)]) +
          (nu * del_t / (del_y * del_y)) * (u[idx(i + 1, j)] + u[idx(i - 1, j)] - 2 * u[idx(i, j)]) -
          (del_t / del_x) * u[idx(i, j)] * (u[idx(i, j)] - u[idx(i, j - 1)]) -
          (del_t / del_y) * v[idx(i, j)] * (u[idx(i, j)] - u[idx(i - 1, j)]);
        v_new[idx(i, j)] = v[idx(i, j)] +
          (nu * del_t / (del_x * del_x)) * (v[idx(i, j + 1)] + v[idx(i, j - 1)] - 2 * v[idx(i, j)]) +
          (nu * del_t / (del_y * del_y)) * (v[idx(i + 1, j)] + v[idx(i - 1, j)] - 2 * v[idx(i, j)]) -
          (del_t / del_x) * u[idx(i, j)] * (v[idx(i, j)] - v[idx(i, j - 1)]) -
          (del_t / del_y) * v[idx(i, j)] * (v[idx(i, j)] - v[idx(i - 1, j)]);
      }
    }
    for (int i = 0; i < x_points; i++) {
      u_new[idx(0, i)] = 1.0;
      v_new[idx(0, i)] = 1.0;
      u_new[idx(y_points - 1, i)] = 1.0;
      v_new[idx(y_points - 1, i)] = 1.0;
    }
    for (int j = 0; j < y_points; j++) {
      u_new[idx(j, 0)] = 1.0;
      v_new[idx(j, 0)] = 1.0;
      u_new[idx(j, x_points - 1)] = 1.0;
      v_new[idx(j, x_points - 1)] = 1.0;
    }
    for (int i = 0; i < y_points; i++) {
      for (int j = 0; j < x_points; j++) {
        u[idx(i, j)] = u_new[idx(i, j)];
        v[idx(i, j)] = v_new[idx(i, j)];
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <dim_x> <dim_y> <nt>\n", argv[0]);
    std::printf("dim_x: number of grid points in the x axis\n");
    std::printf("dim_y: number of grid points in the y axis\n");
    std::printf("nt: number of time steps\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int x_points = std::atoi(argv[1]);
  const int y_points = std::atoi(argv[2]);
  const int num_itrs = std::atoi(argv[3]);
  const double del_x = 2.0 / (x_points - 1);
  const double del_y = 2.0 / (y_points - 1);
  const int grid_elems = x_points * y_points;
  const int grid_size = sizeof(double) * grid_elems;
  const double nu = 0.01;
  const double del_t = 0.0009 * del_x * del_y / nu;

  std::printf("2D Burger's equation\n");
  std::printf("Grid dimension: x = %d y = %d\n", x_points, y_points);
  std::printf("Number of time steps: %d\n", num_itrs);

  double* x = static_cast<double*>(std::malloc(sizeof(double) * x_points));
  double* y = static_cast<double*>(std::malloc(sizeof(double) * y_points));
  double* u = static_cast<double*>(std::malloc(grid_size));
  double* v = static_cast<double*>(std::malloc(grid_size));
  double* u_new = static_cast<double*>(std::malloc(grid_size));
  double* v_new = static_cast<double*>(std::malloc(grid_size));
  double* du = static_cast<double*>(std::malloc(grid_size));
  double* dv = static_cast<double*>(std::malloc(grid_size));
  if (!x || !y || !u || !v || !u_new || !v_new || !du || !dv) {
    std::fprintf(stderr, "allocation failure\n");
    platform.finalize();
    return 1;
  }

  initialize(x, y, u, v, u_new, v_new, x_points, y_points, del_x, del_y);

  iris::DMem mem_u(u, grid_size);
  iris::DMem mem_v(v, grid_size);
  iris::DMem mem_u_new(u_new, grid_size);
  iris::DMem mem_v_new(v_new, grid_size);

  const size_t core_gws[2] = {
    static_cast<size_t>(((x_points - 2 + 15) / 16) * 16),
    static_cast<size_t>(((y_points - 2 + 15) / 16) * 16)
  };
  const size_t core_lws[2] = {16, 16};
  const size_t x_gws[1] = {static_cast<size_t>(((x_points + 255) / 256) * 256)};
  const size_t y_gws[1] = {static_cast<size_t>(((y_points + 255) / 256) * 256)};
  const size_t n_gws[1] = {static_cast<size_t>(((grid_elems + 255) / 256) * 256)};
  const size_t one_lws[1] = {256};

  const auto start = std::chrono::steady_clock::now();
  for (int itr = 0; itr < num_itrs; itr++) {
    void* core_params[10] = {&mem_u_new, &mem_v_new, &mem_u, &mem_v,
                             const_cast<int*>(&x_points), const_cast<int*>(&y_points),
                             const_cast<double*>(&nu), const_cast<double*>(&del_t),
                             const_cast<double*>(&del_x), const_cast<double*>(&del_y)};
    int core_pinfo[10] = {iris_w, iris_w, iris_r, iris_r, sizeof(x_points), sizeof(y_points),
                          sizeof(nu), sizeof(del_t), sizeof(del_x), sizeof(del_y)};
    iris::Task core_task;
    core_task.kernel("core", 2, nullptr, const_cast<size_t*>(core_gws),
                     const_cast<size_t*>(core_lws), 10, core_params, core_pinfo);
    core_task.submit(iris_gpu, nullptr, 1);

    void* bh_params[4] = {&mem_u_new, &mem_v_new, const_cast<int*>(&x_points), const_cast<int*>(&y_points)};
    int bh_pinfo[4] = {iris_rw, iris_rw, sizeof(x_points), sizeof(y_points)};
    iris::Task bh_task;
    bh_task.kernel("bound_h", 1, nullptr, const_cast<size_t*>(x_gws),
                   const_cast<size_t*>(one_lws), 4, bh_params, bh_pinfo);
    bh_task.submit(iris_gpu, nullptr, 1);

    void* bv_params[4] = {&mem_u_new, &mem_v_new, const_cast<int*>(&x_points), const_cast<int*>(&y_points)};
    int bv_pinfo[4] = {iris_rw, iris_rw, sizeof(x_points), sizeof(y_points)};
    iris::Task bv_task;
    bv_task.kernel("bound_v", 1, nullptr, const_cast<size_t*>(y_gws),
                   const_cast<size_t*>(one_lws), 4, bv_params, bv_pinfo);
    bv_task.submit(iris_gpu, nullptr, 1);

    void* update_params[5] = {&mem_u, &mem_v, &mem_u_new, &mem_v_new, const_cast<int*>(&grid_elems)};
    int update_pinfo[5] = {iris_w, iris_w, iris_r, iris_r, sizeof(grid_elems)};
    iris::Task update_task;
    update_task.kernel("update", 1, nullptr, const_cast<size_t*>(n_gws),
                       const_cast<size_t*>(one_lws), 5, update_params, update_pinfo);
    update_task.submit(iris_gpu, nullptr, 1);
  }
  iris::Task flush;
  flush.flush_out(mem_u);
  flush.flush_out(mem_v);
  flush.submit(iris_gpu, nullptr, 1);
  const auto end = std::chrono::steady_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Total kernel execution time %f (s)\n", time * 1e-9f);

  for (int i = 0; i < grid_elems; i++) {
    du[i] = u[i];
    dv[i] = v[i];
  }

  std::printf("Serial computing for verification...\n");
  initialize(x, y, u, v, u_new, v_new, x_points, y_points, del_x, del_y);
  serial_reference(u, v, u_new, v_new, x_points, y_points, num_itrs, nu, del_t, del_x, del_y);

  bool ok = true;
  for (int i = 0; i < y_points; i++) {
    for (int j = 0; j < x_points; j++) {
      if (std::fabs(du[idx(i, j)] - u[idx(i, j)]) > 1e-6 ||
          std::fabs(dv[idx(i, j)] - v[idx(i, j)]) > 1e-6) {
        ok = false;
      }
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(x); std::free(y); std::free(u); std::free(v);
  std::free(du); std::free(dv); std::free(u_new); std::free(v_new);
  platform.finalize();
  return ok ? 0 : 1;
}
