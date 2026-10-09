#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../bilateral-cuda/reference.h"

static bool run_filter(const char* kernel_name, const char* label, iris::DMem& src,
                       iris::DMem& dst, float* h_src, float* h_dst, float* r_dst,
                       int w, int h, float a_square, float variance_i,
                       float variance_spatial, int repeat)
{
  const size_t gws[2] = {static_cast<size_t>(((w + 15) / 16) * 16),
                         static_cast<size_t>(((h + 15) / 16) * 16)};
  const size_t lws[2] = {16, 16};
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[7] = {&src, &dst, &w, &h, &a_square, &variance_i, &variance_spatial};
    int pinfo[7] = {iris_r, iris_w, sizeof(w), sizeof(h), sizeof(a_square),
                    sizeof(variance_i), sizeof(variance_spatial)};
    task.kernel(kernel_name, 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                7, params, pinfo);
    if (i == repeat - 1) task.flush_out(dst);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time (%s) %f (ms)\n", label,
              (time * 1e-6f) / repeat);
  if (label[0] == '3') reference<3>(h_src, r_dst, w, h, a_square, variance_i, variance_spatial);
  if (label[0] == '6') reference<6>(h_src, r_dst, w, h, a_square, variance_i, variance_spatial);
  if (label[0] == '9') reference<9>(h_src, r_dst, w, h, a_square, variance_i, variance_spatial);
  for (int i = 0; i < w * h; i++) {
    if (std::fabs(r_dst[i] - h_dst[i]) > 1e-3f) return false;
  }
  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    std::printf("Usage: %s <image width> <image height> <intensity> <spatial> <repeat>\n",
                argv[0]);
    return 1;
  }
  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  const int w = std::atoi(argv[1]);
  const int h = std::atoi(argv[2]);
  const int img_size = w * h;
  const float variance_i = std::atof(argv[3]);
  const float variance_spatial = std::atof(argv[4]);
  const float a_square = 0.5f / (variance_i * static_cast<float>(M_PI));
  const int repeat = std::atoi(argv[5]);
  if (w <= 0 || h <= 0 || variance_i <= 0.0f || variance_spatial <= 0.0f || repeat <= 0) {
    std::printf("Invalid inputs\n");
    platform.finalize();
    return 1;
  }
  const size_t bytes = static_cast<size_t>(img_size) * sizeof(float);
  float* h_src = static_cast<float*>(std::malloc(bytes));
  float* h_dst = static_cast<float*>(std::malloc(bytes));
  float* r_dst = static_cast<float*>(std::malloc(bytes));
  if (!h_src || !h_dst || !r_dst) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(r_dst);
    std::free(h_dst);
    std::free(h_src);
    platform.finalize();
    return 1;
  }
  std::srand(123);
  for (int i = 0; i < img_size; i++) h_src[i] = static_cast<float>(std::rand() % 256);
  iris::DMem src(h_src, bytes);
  iris::DMem dst(h_dst, bytes);
  bool ok = true;
  ok = run_filter("bilateral_filter_r3", "3x3", src, dst, h_src, h_dst, r_dst, w, h,
                  a_square, variance_i, variance_spatial, repeat) && ok;
  ok = run_filter("bilateral_filter_r6", "6x6", src, dst, h_src, h_dst, r_dst, w, h,
                  a_square, variance_i, variance_spatial, repeat) && ok;
  ok = run_filter("bilateral_filter_r9", "9x9", src, dst, h_src, h_dst, r_dst, w, h,
                  a_square, variance_i, variance_spatial, repeat) && ok;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  std::free(r_dst);
  std::free(h_dst);
  std::free(h_src);
  platform.finalize();
  return ok ? 0 : 1;
}
