#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "reference.h"

static double* generateGaborKernelDevice(
  const int repeat,
  const unsigned int height,
  const unsigned int width,
  const unsigned int par_T,
  const double par_L,
  const double theta)
{
  const double sx = (double)par_T / (2.0*std::sqrt(2.0*std::log(2.0)));
  const double sy = par_L * sx;
  const double sx_2 = sx*sx;
  const double sy_2 = sy*sy;
  const double fx = 1.0 / (double)par_T;
  const double ctheta = std::cos(theta);
  const double stheta = std::sin(theta);
  const double center_y = (double)height / 2.0;
  const double center_x = (double)width / 2.0;
  const double scale = 1.0/(2.0*M_PI*sx*sy);

  size_t image_size_bytes = (size_t)height * (size_t)width * sizeof(double);
  double *h_gabor_spatial = (double*) std::malloc(image_size_bytes);
  if (!h_gabor_spatial) return nullptr;

  iris::DMem d_gabor_spatial(h_gabor_spatial, image_size_bytes);

  size_t gws[2] = {
    ((size_t)width + 15u) / 16u * 16u,
    ((size_t)height + 15u) / 16u * 16u
  };
  size_t lws[2] = {16u, 16u};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[11] = {
      &d_gabor_spatial,
      (void*)&height,
      (void*)&width,
      (void*)&center_y,
      (void*)&center_x,
      (void*)&ctheta,
      (void*)&stheta,
      (void*)&scale,
      (void*)&sx_2,
      (void*)&sy_2,
      (void*)&fx
    };
    int pinfo[11] = {
      iris_w,
      (int)sizeof(height),
      (int)sizeof(width),
      (int)sizeof(center_y),
      (int)sizeof(center_x),
      (int)sizeof(ctheta),
      (int)sizeof(stheta),
      (int)sizeof(scale),
      (int)sizeof(sx_2),
      (int)sizeof(sy_2),
      (int)sizeof(fx)
    };
    task.kernel("gabor", 2, nullptr, gws, lws, 11, params, pinfo);
    if (i == repeat - 1) task.flush_out(d_gabor_spatial);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  return h_gabor_spatial;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::printf("Usage: %s <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int height_arg = std::atoi(argv[1]);
  const int width_arg = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  if (height_arg <= 0 || width_arg <= 0 || repeat <= 0) {
    std::fprintf(stderr, "height, width, and repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  const unsigned int height = (unsigned int)height_arg;
  const unsigned int width = (unsigned int)width_arg;
  const unsigned int par_T = 13;
  const double par_L = 2.65;
  const double theta = 45;

  double *h_filter = generateGaborKernelHost(height, width, par_T, par_L, theta);
  double *d_filter = generateGaborKernelDevice(repeat, height, width, par_T, par_L, theta);
  if (!h_filter || !d_filter) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_filter);
    std::free(d_filter);
    platform.finalize();
    return 1;
  }

  bool ok = true;
  for (unsigned int i = 0; i < width * height; i++) {
    if (std::fabs(h_filter[i] - d_filter[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  std::free(h_filter);
  std::free(d_filter);
  platform.finalize();
  return ok ? 0 : 2;
}
