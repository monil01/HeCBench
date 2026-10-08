#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cooling.hpp"

static void reference(const int num,
                      const Real n,
                      const Real* __restrict__ T,
                      Real* __restrict__ r,
                      const int heat_flag)
{
  for (int i = 0; i < num; i++) {
    r[i] = primordial_cool(n, T[i], heat_flag);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int num = std::atoi(argv[1]);
  int repeat = std::atoi(argv[2]);
  size_t size_bytes = sizeof(Real) * static_cast<size_t>(num);
  Real n = 0.0899;

  Real* T = static_cast<Real*>(std::malloc(size_bytes));
  Real* r = static_cast<Real*>(std::malloc(size_bytes));
  Real* h_r = static_cast<Real*>(std::malloc(size_bytes));
  if (!T || !r || !h_r) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(T);
    std::free(r);
    std::free(h_r);
    platform.finalize();
    return 1;
  }
  for (int i = 0; i < num; i++) {
    T[i] = -275.0 + i * 275 * 2.0 / num;
  }

  iris::DMem d_T(T, size_bytes);
  iris::DMem d_r(r, size_bytes);
  int heat_flag = 1;
  size_t gws[1] = {static_cast<size_t>((num + 255) / 256) * 256};
  size_t lws[1] = {256};

  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[5] = {&num, &n, &d_T, &d_r, &heat_flag};
    int pinfo[5] = {sizeof(num), sizeof(n), iris_r, iris_w, sizeof(heat_flag)};
    task.kernel("cool_kernel", 1, nullptr, gws, lws, 5, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[5] = {&num, &n, &d_T, &d_r, &heat_flag};
    int pinfo[5] = {sizeof(num), sizeof(n), iris_r, iris_w, sizeof(heat_flag)};
    task.kernel("cool_kernel", 1, nullptr, gws, lws, 5, params, pinfo);
    task.flush_out(d_r);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / repeat);

  d_r.fetch(r);
  reference(num, n, T, h_r, heat_flag);

  bool error = false;
  for (int i = 0; i < num; i++) {
    if (std::fabs(r[i] - h_r[i]) > 1e-3) {
      error = true;
      break;
    }
  }
  std::printf("%s\n", error ? "FAIL" : "PASS");

  std::free(T);
  std::free(r);
  std::free(h_r);
  platform.finalize();
  return error ? 1 : 0;
}
