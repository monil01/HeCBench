#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "reference.h"

#define A 0
#define B 15
#define ROW_SIZE 17
#define EPS 1e-7
#define MAX_WORK_GROUP_SIZE 256

static double f_host(double x)
{
  return std::exp(x) * std::sin(x);
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::printf("Usage: %s <number of work-groups> ", argv[0]);
    std::printf("<work-group size> <repeat>\n");
    return 1;
  }

  const int nwg = std::atoi(argv[1]);
  const int wgs = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  if (nwg <= 0 || wgs <= 0 || repeat <= 0 || wgs > MAX_WORK_GROUP_SIZE) {
    std::fprintf(stderr, "Invalid arguments; require positive values and work-group size <= %d\n",
                 MAX_WORK_GROUP_SIZE);
    return 1;
  }

  double* h_result = static_cast<double*>(std::malloc(static_cast<size_t>(nwg) * sizeof(double)));
  if (h_result == nullptr) {
    std::fprintf(stderr, "Host memory allocation failed\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem result_mem(h_result, static_cast<size_t>(nwg) * sizeof(double));
  const double a = A;
  const double b = B;
  const size_t gws[1] = {static_cast<size_t>(nwg) * static_cast<size_t>(wgs)};
  const size_t lws[1] = {static_cast<size_t>(wgs)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = {const_cast<double*>(&a), const_cast<double*>(&b), &result_mem};
    int pinfo[3] = {sizeof(a), sizeof(b), iris_w};
    task.kernel("romberg", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                3, params, pinfo);
    if (i == repeat - 1) task.flush_out(result_mem);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);

  double sum = 0.0;
  for (int k = 0; k < nwg; k++) sum += h_result[k];

  double ref_sum = reference(f_host, A, B, ROW_SIZE, EPS);
  std::printf("%s\n", (std::fabs(sum - ref_sum) > EPS) ? "FAIL" : "PASS");

  platform.finalize();
  std::free(h_result);
  return 0;
}
