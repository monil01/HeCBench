#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "distort_common.h"

static float calc_shift(float x1, float x2, float cx, float k, float thresh)
{
  float x3 = x1 + (x2 - x1) * 0.5f;
  float result1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)));
  float result3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)));

  if (result1 > -thresh && result1 < thresh) {
    return x1;
  }
  if (result3 < 0) {
    return calc_shift(x3, x2, cx, k, thresh);
  }
  return calc_shift(x1, x3, cx, k, thresh);
}

static inline float getRadialX(float x, float y, const Properties* prop)
{
  x = (x * prop->xscale + prop->xshift);
  y = (y * prop->yscale + prop->yshift);
  return x + ((x - prop->centerX) * prop->K *
    ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
}

static inline float getRadialY(float x, float y, const Properties* prop)
{
  x = (x * prop->xscale + prop->xshift);
  y = (y * prop->yscale + prop->yshift);
  return y + ((y - prop->centerY) * prop->K *
    ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
}

static inline void sampleImageTest(const Pixel* src, float idx0, float idx1,
                                   Pixel& result, const Properties* prop)
{
  if ((idx0 < 0) || (idx1 < 0) || (idx0 > prop->height - 1) || (idx1 > prop->width - 1)) {
    result.x = 0;
    result.y = 0;
    result.z = 0;
    return;
  }

  int idx0_floor = static_cast<int>(std::floor(idx0));
  int idx0_ceil = static_cast<int>(std::ceil(idx0));
  int idx1_floor = static_cast<int>(std::floor(idx1));
  int idx1_ceil = static_cast<int>(std::ceil(idx1));

  Pixel s1 = src[(idx0_floor * prop->width) + idx1_floor];
  Pixel s2 = src[(idx0_floor * prop->width) + idx1_ceil];
  Pixel s3 = src[(idx0_ceil * prop->width) + idx1_ceil];
  Pixel s4 = src[(idx0_ceil * prop->width) + idx1_floor];

  float x = idx0 - idx0_floor;
  float y = idx1 - idx1_floor;

  result.x = s1.x * (1.f - x) * (1.f - y) + s2.x * (1.f - x) * y + s3.x * x * y + s4.x * x * (1.f - y);
  result.y = s1.y * (1.f - x) * (1.f - y) + s2.y * (1.f - x) * y + s3.y * x * y + s4.y * x * (1.f - y);
  result.z = s1.z * (1.f - x) * (1.f - y) + s2.z * (1.f - x) * y + s3.z * x * y + s4.z * x * (1.f - y);
}

static void reference(const Pixel* src, Pixel* dst, const Properties* prop)
{
  for (int h = 0; h < prop->height; h++) {
    for (int w = 0; w < prop->width; w++) {
      float x = getRadialX(static_cast<float>(w), static_cast<float>(h), prop);
      float y = getRadialY(static_cast<float>(w), static_cast<float>(h), prop);
      Pixel temp;
      sampleImageTest(src, y, x, temp, prop);
      dst[(h * prop->width) + w] = temp;
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    std::cout << "Usage: " << argv[0]
              << "<input image width> <input image height> <coefficient of distortion> <repeat>\n";
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int width = std::atoi(argv[1]);
  const int height = std::atoi(argv[2]);
  const float K = std::atof(argv[3]);
  const int repeat = std::atoi(argv[4]);

  Properties prop;
  prop.K = K;
  prop.centerX = width / 2;
  prop.centerY = height / 2;
  prop.width = width;
  prop.height = height;
  prop.thresh = 1.f;

  prop.xshift = calc_shift(0, prop.centerX - 1, prop.centerX, prop.K, prop.thresh);
  float newcenterX = prop.width - prop.centerX;
  float xshift_2 = calc_shift(0, newcenterX - 1, newcenterX, prop.K, prop.thresh);

  prop.yshift = calc_shift(0, prop.centerY - 1, prop.centerY, prop.K, prop.thresh);
  float newcenterY = prop.height - prop.centerY;
  float yshift_2 = calc_shift(0, newcenterY - 1, newcenterY, prop.K, prop.thresh);

  prop.xscale = (prop.width - prop.xshift - xshift_2) / prop.width;
  prop.yscale = (prop.height - prop.yshift - yshift_2) / prop.height;

  const int imageSize = height * width;
  const size_t imageSize_bytes = static_cast<size_t>(imageSize) * sizeof(Pixel);
  std::vector<Pixel> h_src(imageSize);
  std::vector<Pixel> h_dst(imageSize);
  std::vector<Pixel> r_dst(imageSize);

  std::srand(123);
  for (int i = 0; i < imageSize; i++) {
    h_src[i].x = std::rand() % 256;
    h_src[i].y = std::rand() % 256;
    h_src[i].z = std::rand() % 256;
  }

  iris::DMem mem_src(h_src.data(), imageSize_bytes);
  iris::DMem mem_dst(h_dst.data(), imageSize_bytes);
  iris::DMem mem_prop(&prop, sizeof(Properties));

  const size_t block_x = 16;
  const size_t block_y = 16;
  const size_t gws[2] = {
    (static_cast<size_t>(width) + block_x - 1) / block_x * block_x,
    (static_cast<size_t>(height) + block_y - 1) / block_y * block_y
  };
  const size_t lws[2] = { block_x, block_y };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = { &mem_src, &mem_dst, &mem_prop };
    int pinfo[3] = { iris_r, iris_w, iris_r };
    task.kernel("barrel_distort", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 3, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_dst);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  reference(h_src.data(), r_dst.data(), &prop);
  int ex = 0;
  int ey = 0;
  int ez = 0;
  for (int i = 0; i < imageSize; i++) {
    ex = std::max(std::abs(static_cast<int>(h_dst[i].x) - static_cast<int>(r_dst[i].x)), ex);
    ey = std::max(std::abs(static_cast<int>(h_dst[i].y) - static_cast<int>(r_dst[i].y)), ey);
    ez = std::max(std::abs(static_cast<int>(h_dst[i].z) - static_cast<int>(r_dst[i].z)), ez);
  }

  std::cout << "Max error of each channel: " << ex << " " << ey << " " << ez << std::endl;

  platform.finalize();
  return (ex <= 1 && ey <= 1 && ez <= 1) ? 0 : 1;
}
