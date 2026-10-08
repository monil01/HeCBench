#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define POLE_REF (std::sqrt(3.0f) - 2.0f)

typedef unsigned int uint;
typedef unsigned char uchar;

static int pow_two_divider(int n)
{
  if (n == 0) return 0;
  int divider = 1;
  while ((n & divider) == 0) divider <<= 1;
  return divider;
}

static float initial_causal_coefficient_ref(float* c, uint data_length, int step)
{
  const uint horizon = (12 < data_length) ? 12 : data_length;
  float zn = POLE_REF;
  float sum = *c;
  for (uint n = 0; n < horizon; n++) {
    sum += zn * (*c);
    zn *= POLE_REF;
    c = reinterpret_cast<float*>(reinterpret_cast<uchar*>(c) + step);
  }
  return sum;
}

static float initial_anti_causal_coefficient_ref(float* c)
{
  return (POLE_REF / (POLE_REF - 1.0f)) * (*c);
}

static void convert_to_interpolation_coefficients_ref(float* coeffs,
                                                      uint data_length,
                                                      int step)
{
  const float lambda = (1.0f - POLE_REF) * (1.0f - 1.0f / POLE_REF);
  float* c = coeffs;
  float previous_c;

  *c = previous_c = lambda * initial_causal_coefficient_ref(c, data_length, step);
  for (uint n = 1; n < data_length; n++) {
    c = reinterpret_cast<float*>(reinterpret_cast<uchar*>(c) + step);
    *c = previous_c = lambda * (*c) + POLE_REF * previous_c;
  }

  *c = previous_c = initial_anti_causal_coefficient_ref(c);
  for (int n = static_cast<int>(data_length) - 2; n >= 0; n--) {
    c = reinterpret_cast<float*>(reinterpret_cast<uchar*>(c) - step);
    *c = previous_c = POLE_REF * (previous_c - (*c));
  }
}

static void to_coef_2d_x_ref(float* image, uint pitch, uint width, uint height)
{
  for (uint y = 0; y < height; y++) {
    float* line = reinterpret_cast<float*>(reinterpret_cast<uchar*>(image) + y * pitch);
    convert_to_interpolation_coefficients_ref(line, width, sizeof(float));
  }
}

static void to_coef_2d_y_ref(float* image, uint pitch, uint width, uint height)
{
  for (uint x = 0; x < width; x++) {
    float* line = image + x;
    convert_to_interpolation_coefficients_ref(line, height, pitch);
  }
}

static void launch_kernel(iris::DMem& image,
                          const char* kernel_name,
                          uint pitch,
                          uint width,
                          uint height,
                          size_t global_size,
                          size_t local_size)
{
  iris::Task task;
  void* params[4] = {&image, &pitch, &width, &height};
  int pinfo[4] = {iris_rw, sizeof(pitch), sizeof(width), sizeof(height)};
  size_t gws[1] = {global_size};
  size_t lws[1] = {local_size};
  task.kernel(kernel_name, 1, nullptr, gws, lws, 4, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const uint width = static_cast<uint>(std::atoi(argv[1]));
  const uint height = static_cast<uint>(std::atoi(argv[2]));
  const int repeat = std::atoi(argv[3]);
  const uint image_pitch = width * sizeof(float);
  const size_t num_pix = static_cast<size_t>(width) * height;

  std::vector<float> image(num_pix);
  std::vector<float> image_ref(num_pix);

  std::mt19937 gen{123};
  std::normal_distribution<float> dist{0.f, 1.f};
  for (uint h = 0; h < height; h++) {
    for (uint w = 0; w < width; w++) {
      image_ref[static_cast<size_t>(h) * width + w] =
          image[static_cast<size_t>(h) * width + w] = dist(gen);
    }
  }

  iris::DMem mem_image(image.data(), image.size() * sizeof(float));
  mem_image.update(image.data());

  int blocks = std::min(pow_two_divider(static_cast<int>(height)), 64);
  const size_t local_x = static_cast<size_t>(blocks);
  const size_t global_x = static_cast<size_t>((height + blocks - 1) / blocks) * blocks;

  blocks = std::min(pow_two_divider(static_cast<int>(width)), 64);
  const size_t local_y = static_cast<size_t>(blocks);
  const size_t global_y = static_cast<size_t>((width + blocks - 1) / blocks) * blocks;

  launch_kernel(mem_image, "toCoef2DX", image_pitch, width, height, global_x, local_x);
  launch_kernel(mem_image, "toCoef2DY", image_pitch, width, height, global_y, local_y);

  iris::Task flush;
  flush.flush_out(mem_image);
  flush.submit(iris_gpu, nullptr, 1);
  mem_image.fetch(image.data());

  to_coef_2d_x_ref(image_ref.data(), image_pitch, width, height);
  to_coef_2d_y_ref(image_ref.data(), image_pitch, width, height);

  bool ok = true;
  for (size_t i = 0; i < num_pix; i++) {
    if (std::fabs(image_ref[i] - image[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_kernel(mem_image, "toCoef2DX", image_pitch, width, height, global_x, local_x);
    launch_kernel(mem_image, "toCoef2DY", image_pitch, width, height, global_y, local_y);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

  platform.finalize();
  return ok ? 0 : 1;
}
