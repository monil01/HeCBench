#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void reference(int n, const int* y, const float* x1, const float* x2,
               const float* d_output, float margin, float* dx1, float* dx2) {
  for (int i = 0; i < n; ++i) {
    float dist = -y[i] * (x1[i] - x2[i]) + margin;
    if (dist < 0.0f) {
      dx1[i] = 0.0f;
      dx2[i] = 0.0f;
    } else {
      dx1[i] = -y[i] * d_output[i];
      dx2[i] = y[i] * d_output[i];
    }
  }
}

bool verify(int n, const std::vector<float>& dx1, const std::vector<float>& dx2,
            const std::vector<float>& ref_dx1, const std::vector<float>& ref_dx2) {
  for (int i = 0; i < n; ++i) {
    if (std::fabs(dx1[i] - ref_dx1[i]) > 1.0e-3f ||
        std::fabs(dx2[i] - ref_dx2[i]) > 1.0e-3f) {
      return false;
    }
  }
  return true;
}

double run_repeated(const char* kernel_name, int repeat, int length, float margin,
                    size_t global_size, size_t local_size, iris::DMem& y_mem,
                    iris::DMem& x1_mem, iris::DMem& x2_mem, iris::DMem& o_mem,
                    iris::DMem& dx1_mem, iris::DMem& dx2_mem,
                    bool flush_outputs) {
  auto start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < repeat; ++iter) {
    void* params[] = {&length, &y_mem, &x1_mem, &x2_mem, &o_mem, &margin, &dx1_mem, &dx2_mem};
    int pinfo[] = {sizeof(length), iris_r, iris_r, iris_r, iris_r, sizeof(margin), iris_w, iris_w};
    size_t gws[] = {global_size};
    size_t lws[] = {local_size};
    iris::Task task;
    task.kernel(kernel_name, 1, nullptr, gws, lws, 8, params, pinfo);
    if (flush_outputs) {
      task.flush_out(dx1_mem);
      task.flush_out(dx2_mem);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1.0e-3 / repeat;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int length = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (length <= 0 || repeat <= 0) {
    std::fprintf(stderr, "length and repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  constexpr float margin = 0.01f;
  constexpr size_t block_size = 256;
  const size_t grid = (static_cast<size_t>(length) + block_size - 1) / block_size;
  const size_t grid2 = (static_cast<size_t>(length / 4) + block_size - 1) / block_size;

  std::vector<float> x1(length), x2(length), output(length), dx1(length), dx2(length);
  std::vector<float> ref_dx1(length), ref_dx2(length);
  std::vector<int> y(length);

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-2.0f, 2.0f);
  for (int i = 0; i < length; ++i) {
    x1[i] = distr(g);
    x2[i] = distr(g);
    output[i] = distr(g);
    y[i] = (distr(g) < 0.0f) ? -1 : 1;
  }

  iris::DMem x1_mem(x1.data(), x1.size() * sizeof(float));
  iris::DMem x2_mem(x2.data(), x2.size() * sizeof(float));
  iris::DMem o_mem(output.data(), output.size() * sizeof(float));
  iris::DMem y_mem(y.data(), y.size() * sizeof(int));
  iris::DMem dx1_mem(dx1.data(), dx1.size() * sizeof(float));
  iris::DMem dx2_mem(dx2.data(), dx2.size() * sizeof(float));

  for (int i = 0; i < repeat; ++i) {
    run_repeated("MRCGradient", 1, length, margin, grid * block_size, block_size,
                 y_mem, x1_mem, x2_mem, o_mem, dx1_mem, dx2_mem, false);
    run_repeated("MRCGradient2", 1, length, margin, grid * block_size, block_size,
                 y_mem, x1_mem, x2_mem, o_mem, dx1_mem, dx2_mem, false);
    run_repeated("MRCGradient3", 1, length, margin, grid2 * block_size, block_size,
                 y_mem, x1_mem, x2_mem, o_mem, dx1_mem, dx2_mem, false);
  }

  double t1 = run_repeated("MRCGradient", repeat, length, margin, grid * block_size,
                           block_size, y_mem, x1_mem, x2_mem, o_mem, dx1_mem, dx2_mem, false);
  std::printf("Average execution time of MRC kernel: %f (us)\n", t1);

  double t2 = run_repeated("MRCGradient2", repeat, length, margin, grid * block_size,
                           block_size, y_mem, x1_mem, x2_mem, o_mem, dx1_mem, dx2_mem, false);
  std::printf("Average execution time of MRC2 kernel: %f (us)\n", t2);

  double t3 = run_repeated("MRCGradient3", repeat, length, margin, grid2 * block_size,
                           block_size, y_mem, x1_mem, x2_mem, o_mem, dx1_mem, dx2_mem, true);
  std::printf("Average execution time of MRC3 kernel: %f (us)\n", t3);

  reference(length, y.data(), x1.data(), x2.data(), output.data(), margin, ref_dx1.data(), ref_dx2.data());
  std::printf("%s\n", verify(length, dx1, dx2, ref_dx1, ref_dx2) ? "PASS" : "FAIL");

  platform.finalize();
  return 0;
}
