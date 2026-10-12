#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#define nthreads 256

struct float3 {
  float x, y, z;
};

static void run_kernel_pair(float3* h_img, unsigned int* h_tmp, unsigned int* h_out,
                            int width, int height, size_t image_bytes,
                            size_t output_bytes)
{
  iris::DMem d_img(h_img, image_bytes);
  iris::DMem d_tmp(h_tmp, output_bytes);
  iris::DMem d_out(h_out, output_bytes);

  int w = width;
  int h = height;
  size_t gws[1] = {static_cast<size_t>(width * height)};
  size_t lws[1] = {nthreads};

  void* params1[4] = {&d_img, &d_tmp, &w, &h};
  int pinfo1[4] = {iris_r, iris_w, sizeof(w), sizeof(h)};
  iris::Task task1;
  task1.kernel("check_connect", 1, nullptr, gws, lws, 4, params1, pinfo1);
  task1.submit(iris_gpu, nullptr, 1);

  void* params2[4] = {&d_tmp, &d_out, &w, &h};
  int pinfo2[4] = {iris_r, iris_w, sizeof(w), sizeof(h)};
  iris::Task task2;
  task2.kernel("eliminate_crosses", 1, nullptr, gws, lws, 4, params2, pinfo2);
  task2.flush_out(d_out);
  task2.submit(iris_gpu, nullptr, 1);
  d_out.fetch(h_out);
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  int size = width * height;
  size_t size_output_bytes = size * sizeof(unsigned int);
  size_t size_image_bytes = size * sizeof(float3);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.f, 0.4f);

  float3* h_img = (float3*)malloc(size_image_bytes);
  unsigned int* h_tmp = (unsigned int*)malloc(size_output_bytes);
  unsigned int* h_out = (unsigned int*)malloc(size_output_bytes);

  float sum = 0;
  float total_time = 0;

  for (int n = 0; n < repeat; n++) {
    for (int i = 0; i < size; i++) {
      h_img[i].x = dis(gen);
      h_img[i].y = dis(gen);
      h_img[i].z = dis(gen);
    }

    auto start = std::chrono::steady_clock::now();
    run_kernel_pair(h_img, h_tmp, h_out, width, height, size_image_bytes, size_output_bytes);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<float> time = end - start;
    total_time += time.count();

    float lsum = 0;
    for (int i = 0; i < size; i++) {
      lsum += (h_out[i] & 0xff) / 256.f + ((h_out[i] >> 8) & 0xff) / 256.f +
              ((h_out[i] >> 16) & 0xff) / 256.f + ((h_out[i] >> 24) & 0xff) / 256.f;
    }

    sum += lsum / size;
  }

  printf("Image size: %d (width) x %d (height)\ncheckSum: %f\n", width, height, sum);
  printf("Average kernel time over %d iterations: %f (s)\n", repeat, total_time / repeat);

  free(h_out);
  free(h_tmp);
  free(h_img);
  platform.finalize();
  return 0;
}
