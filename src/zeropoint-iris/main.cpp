#include <iris/iris.hpp>

#include "reference.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <number of min/max values> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int size = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  const int32_t qmin = -127;
  const int32_t qmax = 127;
  const bool preserve_sparsity = true;
  const size_t float_bytes = sizeof(float) * static_cast<size_t>(size);
  const size_t int_bytes = sizeof(int32_t) * static_cast<size_t>(size);

  float* scale = static_cast<float*>(std::malloc(float_bytes));
  float* scale_ref = static_cast<float*>(std::malloc(float_bytes));
  int32_t* zp = static_cast<int32_t*>(std::malloc(int_bytes));
  int32_t* zp_ref = static_cast<int32_t*>(std::malloc(int_bytes));
  float* min = static_cast<float*>(std::malloc(float_bytes));
  float* max = static_cast<float*>(std::malloc(float_bytes));
  if (!scale || !scale_ref || !zp || !zp_ref || !min || !max) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(scale);
    std::free(scale_ref);
    std::free(zp);
    std::free(zp_ref);
    std::free(min);
    std::free(max);
    platform.finalize();
    return 1;
  }

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-1.f, 1.f);
  for (int i = 0; i < size; i++) {
    min[i] = distr(g);
    max[i] = distr(g);
    scale[i] = 0.0f;
    zp[i] = 0;
  }

  reference(min, max, qmin, qmax, size, preserve_sparsity, scale_ref, zp_ref);

  iris::DMem mem_min(min, float_bytes);
  iris::DMem mem_max(max, float_bytes);
  iris::DMem mem_scale(scale, float_bytes);
  iris::DMem mem_zp(zp, int_bytes);

  constexpr int block_size = 256;
  size_t gws[1] = {
      static_cast<size_t>((size + block_size - 1) / block_size * block_size)};
  size_t lws[1] = {block_size};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[8] = {
        &mem_min,
        &mem_max,
        const_cast<int32_t*>(&qmin),
        const_cast<int32_t*>(&qmax),
        const_cast<int*>(&size),
        const_cast<bool*>(&preserve_sparsity),
        &mem_scale,
        &mem_zp};
    int pinfo[8] = {
        iris_r,
        iris_r,
        sizeof(qmin),
        sizeof(qmax),
        sizeof(size),
        sizeof(preserve_sparsity),
        iris_w,
        iris_w};
    task.kernel("zero_point", 1, nullptr, gws, lws, 8, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_scale);
      task.flush_out(mem_zp);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of zero-point kernel: %f (us)\n",
              (time * 1e-3f) / repeat);

  mem_scale.fetch(scale);
  mem_zp.fetch(zp);

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (zp[i] != zp_ref[i] || scale[i] - scale_ref[i] > 1e-3f) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(zp);
  std::free(scale);
  std::free(zp_ref);
  std::free(scale_ref);
  std::free(min);
  std::free(max);

  platform.finalize();
  return ok ? 0 : 2;
}
