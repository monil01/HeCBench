#include <iris/iris.hpp>

#include "atan2_common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

template <typename T>
static float rmse_integral(const int n, const T* ref, const T* out)
{
  float error = 0;
  for (int i = 0; i < n; i++) {
    if (std::abs(ref[i] - out[i]) > 0) {
      const float diff = static_cast<float>(ref[i] - out[i]);
      error += diff * diff;
    }
  }
  return std::sqrt(error / n);
}

static float rmse_float(const int n, const float* ref, const float* out)
{
  float error = 0;
  for (int i = 0; i < n; i++) {
    if (std::fabs(ref[i] - out[i]) > 1e-3f) {
      const float diff = ref[i] - out[i];
      error += diff * diff;
    }
  }
  return std::sqrt(error / n);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of coordinates> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  const size_t input_bytes = sizeof(float) * n;
  const size_t output_float_bytes = sizeof(float) * n;
  const size_t output_int_bytes = sizeof(int) * n;
  const size_t output_short_bytes = sizeof(short) * n;

  float *x = static_cast<float*>(std::malloc(input_bytes));
  float *y = static_cast<float*>(std::malloc(input_bytes));
  float *hf = static_cast<float*>(std::malloc(output_float_bytes));
  int *hi = static_cast<int*>(std::malloc(output_int_bytes));
  short *hs = static_cast<short*>(std::malloc(output_short_bytes));
  float *rf = static_cast<float*>(std::malloc(output_float_bytes));
  int *ri = static_cast<int*>(std::malloc(output_int_bytes));
  short *rs = static_cast<short*>(std::malloc(output_short_bytes));
  if (!x || !y || !hf || !hi || !hs || !rf || !ri || !rs) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(x); std::free(y); std::free(hf); std::free(hi);
    std::free(hs); std::free(rf); std::free(ri); std::free(rs);
    platform.finalize();
    return 1;
  }

  srand(123);
  for (int i = 0; i < n; i++) {
    x[i] = rand() / static_cast<float>(RAND_MAX) + 1.57f;
    y[i] = rand() / static_cast<float>(RAND_MAX) + 1.57f;
  }

  iris::DMem mem_x(x, input_bytes);
  iris::DMem mem_y(y, input_bytes);
  iris::DMem mem_f(hf, output_float_bytes);
  iris::DMem mem_i(hi, output_int_bytes);
  iris::DMem mem_s(hs, output_short_bytes);

  size_t gws[1] = {static_cast<size_t>(n / 256 + 1) * 256};
  size_t lws[1] = {256};

  std::printf("\n======== output type is f32 ========\n");
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {const_cast<int*>(&n), &mem_y, &mem_x, &mem_f};
    int pinfo[4] = {sizeof(n), iris_r, iris_r, iris_w};
    task.kernel("compute_f", 1, nullptr, gws, lws, 4, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_f);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);
  mem_f.fetch(hf);
  reference_f(n, y, x, rf);
  std::printf("RMSE: %f\n", rmse_float(n, rf, hf));

  std::printf("\n======== output type is i32 ========\n");
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {const_cast<int*>(&n), &mem_y, &mem_x, &mem_i};
    int pinfo[4] = {sizeof(n), iris_r, iris_r, iris_w};
    task.kernel("compute_i", 1, nullptr, gws, lws, 4, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_i);
    task.submit(iris_gpu, nullptr, 1);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);
  mem_i.fetch(hi);
  reference_i(n, y, x, ri);
  std::printf("RMSE: %f\n", rmse_integral(n, ri, hi));

  std::printf("\n======== output type is i16 ========\n");
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {const_cast<int*>(&n), &mem_y, &mem_x, &mem_s};
    int pinfo[4] = {sizeof(n), iris_r, iris_r, iris_w};
    task.kernel("compute_s", 1, nullptr, gws, lws, 4, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_s);
    task.submit(iris_gpu, nullptr, 1);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);
  mem_s.fetch(hs);
  reference_s(n, y, x, rs);
  std::printf("RMSE: %f\n", rmse_integral(n, rs, hs));

  std::free(x); std::free(y); std::free(hf); std::free(hi);
  std::free(hs); std::free(rf); std::free(ri); std::free(rs);
  platform.finalize();
  return 0;
}
