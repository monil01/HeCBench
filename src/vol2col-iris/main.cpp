#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "reference.h"

static constexpr int threadsPerBlock = 512;

static int get_blocks(int64_t n) {
  int64_t num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  return static_cast<int>(std::min<int64_t>(2147483647, num_blocks));
}

static double launch_vol2col(
    int repeat, int64_t n, float* data_vol, float* data_col, int depth, int height,
    int width, int ksize_t, int ksize_h, int ksize_w, int pad_t, int pad_h, int pad_w,
    int stride_t, int stride_h, int stride_w, int dilation_t, int dilation_h,
    int dilation_w, int depth_col, int height_col, int width_col, size_t vol_size_bytes,
    size_t col_size_bytes) {
  int blocks = get_blocks(n);
  size_t gws[] = {static_cast<size_t>(blocks) * threadsPerBlock};
  size_t lws[] = {threadsPerBlock};
  iris::DMem mem_vol(data_vol, vol_size_bytes);
  iris::DMem mem_col(data_col, col_size_bytes);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[] = {&n, &mem_vol, &depth, &height, &width, &ksize_t, &ksize_h,
                      &ksize_w, &pad_t, &pad_h, &pad_w, &stride_t, &stride_h,
                      &stride_w, &dilation_t, &dilation_h, &dilation_w, &depth_col,
                      &height_col, &width_col, &mem_col};
    int pinfo[] = {static_cast<int>(sizeof(n)), iris_r,
                   static_cast<int>(sizeof(depth)), static_cast<int>(sizeof(height)),
                   static_cast<int>(sizeof(width)), static_cast<int>(sizeof(ksize_t)),
                   static_cast<int>(sizeof(ksize_h)), static_cast<int>(sizeof(ksize_w)),
                   static_cast<int>(sizeof(pad_t)), static_cast<int>(sizeof(pad_h)),
                   static_cast<int>(sizeof(pad_w)), static_cast<int>(sizeof(stride_t)),
                   static_cast<int>(sizeof(stride_h)), static_cast<int>(sizeof(stride_w)),
                   static_cast<int>(sizeof(dilation_t)), static_cast<int>(sizeof(dilation_h)),
                   static_cast<int>(sizeof(dilation_w)), static_cast<int>(sizeof(depth_col)),
                   static_cast<int>(sizeof(height_col)), static_cast<int>(sizeof(width_col)),
                   iris_rw};
    task.kernel("vol2col_kernel_float", 1, nullptr, gws, lws, 21, params, pinfo);
    task.flush_out(mem_col);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() *
         1e-3 / repeat;
}

static double launch_col2vol(
    int repeat, int64_t n, float* data_col, float* data_vol, int depth, int height,
    int width, int kernel_t, int kernel_h, int kernel_w, int pad_t, int pad_h, int pad_w,
    int stride_t, int stride_h, int stride_w, int dilation_t, int dilation_h,
    int dilation_w, int depth_col, int height_col, int width_col, size_t col_size_bytes,
    size_t vol_size_bytes) {
  int blocks = get_blocks(n);
  size_t gws[] = {static_cast<size_t>(blocks) * threadsPerBlock};
  size_t lws[] = {threadsPerBlock};
  iris::DMem mem_col(data_col, col_size_bytes);
  iris::DMem mem_vol(data_vol, vol_size_bytes);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[] = {&n, &mem_col, &depth, &height, &width, &kernel_t, &kernel_h,
                      &kernel_w, &pad_t, &pad_h, &pad_w, &stride_t, &stride_h,
                      &stride_w, &dilation_t, &dilation_h, &dilation_w, &depth_col,
                      &height_col, &width_col, &mem_vol};
    int pinfo[] = {static_cast<int>(sizeof(n)), iris_r,
                   static_cast<int>(sizeof(depth)), static_cast<int>(sizeof(height)),
                   static_cast<int>(sizeof(width)), static_cast<int>(sizeof(kernel_t)),
                   static_cast<int>(sizeof(kernel_h)), static_cast<int>(sizeof(kernel_w)),
                   static_cast<int>(sizeof(pad_t)), static_cast<int>(sizeof(pad_h)),
                   static_cast<int>(sizeof(pad_w)), static_cast<int>(sizeof(stride_t)),
                   static_cast<int>(sizeof(stride_h)), static_cast<int>(sizeof(stride_w)),
                   static_cast<int>(sizeof(dilation_t)), static_cast<int>(sizeof(dilation_h)),
                   static_cast<int>(sizeof(dilation_w)), static_cast<int>(sizeof(depth_col)),
                   static_cast<int>(sizeof(height_col)), static_cast<int>(sizeof(width_col)),
                   iris_rw};
    task.kernel("col2vol_kernel_float", 1, nullptr, gws, lws, 21, params, pinfo);
    task.flush_out(mem_vol);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() *
         1e-3 / repeat;
}

static void eval(int repeat, int channels, int depth, int height, int width,
                 int depth_col, int height_col, int width_col, int ksize_t,
                 int ksize_h, int ksize_w, int pad_t, int pad_h, int pad_w,
                 int stride_t, int stride_h, int stride_w, int dilation_t,
                 int dilation_h, int dilation_w) {
  int64_t vol_size = static_cast<int64_t>(channels) * (2 * pad_t + depth) *
                     (2 * pad_h + height) * (2 * pad_w + width);
  int64_t col_size = (static_cast<int64_t>(channels) * ksize_t * ksize_h * ksize_w + 1) *
                     (depth_col + pad_t) * (height_col + pad_h) * (width_col + pad_w);

  size_t vol_size_bytes = static_cast<size_t>(vol_size) * sizeof(float);
  size_t col_size_bytes = static_cast<size_t>(col_size) * sizeof(float);

  float* h_data_vol = static_cast<float*>(malloc(vol_size_bytes));
  float* h_data_col = static_cast<float*>(malloc(col_size_bytes));
  float* h_data_vol_ref = static_cast<float*>(malloc(vol_size_bytes));
  float* h_data_col_ref = static_cast<float*>(malloc(col_size_bytes));
  if (!h_data_vol || !h_data_col || !h_data_vol_ref || !h_data_col_ref) {
    fprintf(stderr, "host allocation failed\n");
    exit(1);
  }

  srand(123);
  for (int64_t i = 0; i < vol_size; i++) {
    h_data_vol_ref[i] = h_data_vol[i] = rand() / static_cast<float>(RAND_MAX);
  }
  memset(h_data_col, 0, col_size_bytes);

  int64_t n = static_cast<int64_t>(channels) * depth_col * height_col * width_col;

  double time = launch_vol2col(repeat, n, h_data_vol, h_data_col, depth, height, width,
                               ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w,
                               stride_t, stride_h, stride_w, dilation_t, dilation_h,
                               dilation_w, depth_col, height_col, width_col,
                               vol_size_bytes, col_size_bytes);
  printf("Average execution time of vol2col kernel: %f (us)\n", time);

  vol2col_reference<float>(h_data_vol, channels, depth, height, width, ksize_t, ksize_h,
                           ksize_w, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w,
                           dilation_t, dilation_h, dilation_w, depth_col, height_col,
                           width_col, h_data_col_ref);
  int error = memcmp(h_data_col_ref, h_data_col, col_size_bytes);
  printf("%s\n", error ? "FAIL" : "PASS");

  time = launch_col2vol(repeat, n, h_data_col, h_data_vol, depth, height, width,
                        ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w, stride_t,
                        stride_h, stride_w, dilation_t, dilation_h, dilation_w,
                        depth_col, height_col, width_col, col_size_bytes,
                        vol_size_bytes);
  printf("Average execution time of col2vol kernel: %f (us)\n", time);

  col2vol_reference<float, float>(h_data_col_ref, channels, depth, height, width,
                                  ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w,
                                  stride_t, stride_h, stride_w, dilation_t, dilation_h,
                                  dilation_w, depth_col, height_col, width_col,
                                  h_data_vol_ref);
  for (int64_t i = 0; i < vol_size; i++) {
    if (std::fabs(h_data_vol_ref[i] - h_data_vol[i]) > 1e-3f) {
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  free(h_data_vol);
  free(h_data_col);
  free(h_data_vol_ref);
  free(h_data_col_ref);
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    platform.finalize();
    return 1;
  }
  const int repeat = atoi(argv[1]);
  if (repeat <= 0) {
    fprintf(stderr, "repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  int channels = 4;
  int depth = 3;
  int height = 255;
  int width = 255;
  int pad_t = 1;
  int pad_h = 1;
  int pad_w = 1;
  int stride_t = 2;
  int stride_h = 2;
  int stride_w = 2;
  int dilation_t = 2;
  int dilation_h = 2;
  int dilation_w = 2;
  int depth_col = 3;
  int height_col = 255;
  int width_col = 255;

  for (int k = 1; k <= 9; k += 2) {
    printf("\nkernel size: %d\n", k);
    eval(repeat, channels, depth, height, width, depth_col, height_col, width_col,
         k, k, k, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t,
         dilation_h, dilation_w);
  }

  platform.finalize();
  return 0;
}
