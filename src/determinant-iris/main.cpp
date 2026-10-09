#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int Nrows = 11;
  const int Ncols = 11;
  float h_A[Nrows * Ncols] = {
      2.,  -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
      -2., 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
      -2., 0.,  6.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
      -2., 0.,  2.,  8.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
      -2., 0.,  2.,  4.,  10., 10., 10., 10., 10., 10., 10.,
      -2., 0.,  2.,  4.,  10., 12., 12., 12., 12., 12., 12.,
      -2., 0.,  2.,  4.,  10., 12., 14., 14., 14., 14., 14.,
      -2., 0.,  2.,  4.,  10., 12., 14., 16., 16., 16., 16.,
      -2., 0.,  2.,  4.,  10., 12., 14., 16., 18., 18., 18.,
      -2., 0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 20.,
      -2., 0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 22.};
  float det = 0.0f;

  iris::DMem mem_A(h_A, sizeof(h_A));
  iris::DMem mem_det(&det, sizeof(det));
  size_t gws = 1;
  size_t lws = 1;
  void* params[] = {&mem_A, &mem_det, (void*)&Nrows};
  int pinfo[] = {iris_r, iris_w, sizeof(Nrows)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    task.kernel("determinant_cholesky", 1, nullptr, &gws, &lws, 3, params, pinfo);
    task.flush_out(mem_det);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

  std::printf("determinant = %f\n", det);
  std::printf("%s\n", (std::fabs(det - 2048.0f) < 0.02f) ? "PASS" : "FAIL");

  platform.finalize();
  return (std::fabs(det - 2048.0f) < 0.02f) ? 0 : 1;
}
