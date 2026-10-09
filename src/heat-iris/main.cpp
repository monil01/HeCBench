#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define LINE "--------------------"

static double solution(const double t, const double x, const double y, const double alpha, const double length)
{
  return std::exp(-2.0 * alpha * M_PI * M_PI * t / (length * length)) *
         std::sin(M_PI * x / length) * std::sin(M_PI * y / length);
}

static double l2norm(const int n, const double* u, const int nsteps, const double dt,
                     const double alpha, const double dx, const double length)
{
  const double time = dt * static_cast<double>(nsteps);
  double norm = 0.0;
  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx;
    for (int i = 0; i < n; ++i) {
      const double answer = solution(time, x, y, alpha, length);
      const double diff = u[i + j * n] - answer;
      norm += diff * diff;
      x += dx;
    }
    y += dx;
  }
  return std::sqrt(norm);
}

static void launch_unary(const char* kernel, iris::DMem& mem, int n, double dx, double length,
                         size_t* gws, size_t* lws)
{
  iris::Task task;
  void* params[4] = { &n, &dx, &length, &mem };
  int pinfo[4] = { sizeof(n), sizeof(dx), sizeof(length), iris_w };
  task.kernel(kernel, 1, nullptr, gws, lws, 4, params, pinfo);
  task.flush_out(mem);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  auto start = std::chrono::high_resolution_clock::now();

  int n = 1000;
  int nsteps = 10;
  if (argc == 3) {
    n = std::atoi(argv[1]);
    if (n < 0) {
      std::cerr << "Error: n must be positive" << std::endl;
      return EXIT_FAILURE;
    }
    nsteps = std::atoi(argv[2]);
    if (nsteps < 0) {
      std::cerr << "Error: nsteps must be positive" << std::endl;
      return EXIT_FAILURE;
    }
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const double alpha = 0.1;
  const double length = 1000.0;
  const double dx = length / (n + 1);
  const double dt = 0.5 / nsteps;
  const double r = alpha * dt / (dx * dx);
  const double r2 = 1.0 - 4.0 * r;

  std::cout << std::endl
            << " MMS heat equation" << std::endl << std::endl
            << LINE << std::endl
            << "Problem input" << std::endl << std::endl
            << " Grid size: " << n << " x " << n << std::endl
            << " Cell width: " << dx << std::endl
            << " Grid length: " << length << "x" << length << std::endl
            << std::endl
            << " Alpha: " << alpha << std::endl
            << std::endl
            << " Steps: " << nsteps << std::endl
            << " Total time: " << dt * static_cast<double>(nsteps) << std::endl
            << " Time step: " << dt << std::endl
            << " GPU device: IRIS HIP" << std::endl
            << LINE << std::endl;

  std::cout << "Stability" << std::endl << std::endl;
  std::cout << " r value: " << r << std::endl;
  if (r > 0.5) std::cout << " Warning: unstable" << std::endl;
  std::cout << LINE << std::endl;

  const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
  const size_t bytes = count * sizeof(double);
  double* u = static_cast<double*>(std::malloc(bytes));
  double* u_tmp = static_cast<double*>(std::malloc(bytes));
  if (!u || !u_tmp) {
    std::cerr << "Host allocation failed" << std::endl;
    std::free(u);
    std::free(u_tmp);
    platform.finalize();
    return EXIT_FAILURE;
  }

  iris::DMem mem_u(u, bytes);
  iris::DMem mem_tmp(u_tmp, bytes);
  const size_t block_size = 256;
  size_t gws[1] = { ((count + block_size - 1) / block_size) * block_size };
  size_t lws[1] = { block_size };

  launch_unary("initial_value", mem_u, n, dx, length, gws, lws);
  {
    iris::Task task;
    void* params[2] = { &n, &mem_tmp };
    int pinfo[2] = { sizeof(n), iris_w };
    task.kernel("zero", 1, nullptr, gws, lws, 2, params, pinfo);
    task.flush_out(mem_tmp);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto tic = std::chrono::high_resolution_clock::now();
  bool u_is_current = true;
  for (int t = 0; t < nsteps; ++t) {
    iris::DMem& in = u_is_current ? mem_u : mem_tmp;
    iris::DMem& out = u_is_current ? mem_tmp : mem_u;
    iris::Task task;
    void* params[7] = { &n, const_cast<double*>(&alpha), const_cast<double*>(&dx),
                        const_cast<double*>(&dt), const_cast<double*>(&r),
                        const_cast<double*>(&r2), &in };
    int pinfo[8] = { sizeof(n), sizeof(alpha), sizeof(dx), sizeof(dt), sizeof(r), sizeof(r2), iris_r, iris_w };
    void* params_with_out[8] = { params[0], params[1], params[2], params[3], params[4], params[5], params[6], &out };
    task.kernel("solve", 1, nullptr, gws, lws, 8, params_with_out, pinfo);
    task.flush_out(out);
    task.submit(iris_gpu, nullptr, 1);
    u_is_current = !u_is_current;
  }
  auto toc = std::chrono::high_resolution_clock::now();

  double* current = u_is_current ? u : u_tmp;
  const double norm = l2norm(n, current, nsteps, dt, alpha, dx, length);
  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "Results" << std::endl << std::endl
            << "Error (L2norm): " << norm << std::endl
            << "Solve time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count() << std::endl
            << "Total time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << std::endl
            << "Bandwidth (GB/s): " << 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) /
                  std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count() << std::endl
            << LINE << std::endl;

  std::free(u);
  std::free(u_tmp);
  platform.finalize();
  return 0;
}
