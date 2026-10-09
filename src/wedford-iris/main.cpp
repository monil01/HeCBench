#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <batch_size> <spatial_size> <feature_size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int batch_size = std::atoi(argv[1]);
  int spatial_size = std::atoi(argv[2]);
  int feature_size = std::atoi(argv[3]);
  int repeat = std::atoi(argv[4]);

  const int block_y = 16;
  const int block_x = 32;

  int fs_bytes = feature_size * static_cast<int>(sizeof(float));
  size_t input_size = static_cast<size_t>(batch_size) * spatial_size * feature_size;
  size_t is_bytes = input_size * sizeof(float);

  std::vector<float> input(input_size);
  std::srand(123);
  for (size_t i = 0; i < input_size; i++) {
    input[i] = std::rand() / static_cast<float>(RAND_MAX);
  }

  std::vector<float> mean(static_cast<size_t>(feature_size));
  std::vector<float> var(static_cast<size_t>(feature_size));
  std::vector<float> r_mean(static_cast<size_t>(feature_size));
  std::vector<float> r_var(static_cast<size_t>(feature_size));

  iris::DMem d_input(input.data(), is_bytes);
  iris::DMem d_mean(mean.data(), fs_bytes);
  iris::DMem d_var(var.data(), fs_bytes);

  const size_t gws[2] = {
      static_cast<size_t>(feature_size * block_x),
      static_cast<size_t>(block_y)};
  const size_t lws[2] = {
      static_cast<size_t>(block_x),
      static_cast<size_t>(block_y)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[6] = {
        &d_input, &d_mean, &d_var, &batch_size, &feature_size, &spatial_size};
    int pinfo[6] = {
        iris_r, iris_w, iris_w, sizeof(batch_size), sizeof(feature_size), sizeof(spatial_size)};
    task.kernel("welford_kernel_float", 2, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 6, params, pinfo);
    task.flush_out(d_mean);
    task.flush_out(d_var);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  d_var.fetch(var.data());
  d_mean.fetch(mean.data());

  welford_reference<float, float, float>(
      input.data(), r_mean.data(), r_var.data(), batch_size, feature_size, spatial_size);

  bool ok = true;
  for (int i = 0; i < feature_size; i++) {
    if (std::fabs(var[static_cast<size_t>(i)] - r_var[static_cast<size_t>(i)]) > 1e-3f ||
        std::fabs(mean[static_cast<size_t>(i)] - r_mean[static_cast<size_t>(i)]) > 1e-3f) {
      std::printf("Error at index %d: %f %f %f %f\n", i,
                  var[static_cast<size_t>(i)], r_var[static_cast<size_t>(i)],
                  mean[static_cast<size_t>(i)], r_mean[static_cast<size_t>(i)]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  return ok ? 0 : 1;
}
