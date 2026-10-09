#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char* argv[])
{
  const int n = (argc >= 2) ? std::atoi(argv[1]) : (1 << 22);
  const int repeat = (argc >= 3) ? std::atoi(argv[2]) : 100;
  const float a = 2.0f;

  std::printf("The system supports 1 ns time resolution\n");

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::vector<float> x(static_cast<size_t>(n));
  std::vector<float> y(static_cast<size_t>(n));
  std::vector<float> y_expected(static_cast<size_t>(n));

  for (int i = 0; i < n; i++) {
    x[static_cast<size_t>(i)] = 1.0f + static_cast<float>((i * 2654435761u) & 0xffff) / 65535.0f;
    y[static_cast<size_t>(i)] = 0.5f + static_cast<float>((i * 40503u) & 0xffff) / 65535.0f;
    y_expected[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] + static_cast<float>(repeat) * a * x[static_cast<size_t>(i)];
  }

  iris::DMem mem_x(x.data(), x.size() * sizeof(float));
  iris::DMem mem_y(y.data(), y.size() * sizeof(float));
  mem_x.update(x.data());
  mem_y.update(y.data());

  const size_t block = 256;
  size_t gws[1] = { ((static_cast<size_t>(n) + block - 1) / block) * block };
  size_t lws[1] = { block };

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    iris::Task task;
    void* params[4] = { const_cast<int*>(&n), const_cast<float*>(&a), &mem_x, &mem_y };
    int pinfo[4] = { sizeof(n), sizeof(a), iris_r, iris_rw };
    task.kernel("saxpy_k", 1, nullptr, gws, lws, 4, params, pinfo);
    if (r == repeat - 1) task.flush_out(mem_y);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / repeat;
  std::printf("Average kernel execution time: %.3f (ms)\n", ms);

  mem_y.fetch(y.data());

  float maxabs = 0.0f;
  for (int i = 0; i < n; i++) {
    const float d = std::fabs(y[static_cast<size_t>(i)] - y_expected[static_cast<size_t>(i)]);
    if (d > maxabs) maxabs = d;
  }

  std::printf("max |err| = %g\n", maxabs);
  std::printf("%s\n", (maxabs < 1e-2f) ? "PASS" : "FAIL");

  platform.finalize();
  return (maxabs < 1e-2f) ? 0 : 1;
}
