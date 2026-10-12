#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "reference.h"

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("./%s <image dimension> <threshold> <max box size> <iterations>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int Lx = std::atoi(argv[1]);
  int Ly = Lx;
  const int size = Lx * Ly;
  int Threshold = std::atoi(argv[2]);
  int MaxRad = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  const size_t size_bytes = size * sizeof(float);
  const size_t box_bytes = size * sizeof(int);

  float* img = static_cast<float*>(std::malloc(size_bytes));
  float* norm = static_cast<float*>(std::malloc(size_bytes));
  float* h_norm = static_cast<float*>(std::malloc(size_bytes));
  int* box = static_cast<int*>(std::malloc(box_bytes));
  int* h_box = static_cast<int*>(std::malloc(box_bytes));
  float* out = static_cast<float*>(std::malloc(size_bytes));
  float* h_out = static_cast<float*>(std::malloc(size_bytes));
  if (!img || !norm || !h_norm || !box || !h_box || !out || !h_out) return 1;

  std::srand(123);
  for (int i = 0; i < size; i++) {
    img[i] = std::rand() % 256;
    norm[i] = 0.0f;
    box[i] = 0;
    out[i] = 0.0f;
  }

  const size_t gws[2] = {static_cast<size_t>(((Lx + 15) / 16) * 16),
                         static_cast<size_t>(((Ly + 15) / 16) * 16)};
  const size_t lws[2] = {16, 16};

  double time = 0.0;
  for (int iter = 0; iter < repeat; iter++) {
    for (int i = 0; i < size; i++) {
      norm[i] = 0.0f;
      box[i] = 0;
      out[i] = 0.0f;
    }
    iris::DMem mem_img(img, size_bytes);
    iris::DMem mem_norm(norm, size_bytes);
    iris::DMem mem_box(box, box_bytes);
    iris::DMem mem_out(out, size_bytes);

    auto start = std::chrono::steady_clock::now();

    {
      void* params[7] = {&Lx, &Ly, &Threshold, &MaxRad, &mem_img, &mem_box, &mem_norm};
      int pinfo[7] = {sizeof(Lx), sizeof(Ly), sizeof(Threshold), sizeof(MaxRad), iris_r,
                      iris_w, iris_rw};
      iris::Task task;
      task.kernel("smoothingFilter", 2, nullptr, const_cast<size_t*>(gws),
                  const_cast<size_t*>(lws), 7, params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
    }
    {
      void* params[4] = {&Lx, &Ly, &mem_img, &mem_norm};
      int pinfo[4] = {sizeof(Lx), sizeof(Ly), iris_rw, iris_r};
      iris::Task task;
      task.kernel("normalizeFilter", 2, nullptr, const_cast<size_t*>(gws),
                  const_cast<size_t*>(lws), 4, params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
    }
    {
      void* params[5] = {&Lx, &Ly, &mem_img, &mem_box, &mem_out};
      int pinfo[5] = {sizeof(Lx), sizeof(Ly), iris_r, iris_r, iris_rw};
      iris::Task task;
      task.kernel("outFilter", 2, nullptr, const_cast<size_t*>(gws),
                  const_cast<size_t*>(lws), 5, params, pinfo);
      if (iter == repeat - 1) {
        task.flush_out(mem_norm);
        task.flush_out(mem_box);
        task.flush_out(mem_out);
      }
      task.submit(iris_gpu, nullptr, 1);
    }

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  std::printf("Average filtering time %lf (s)\n", (time * 1e-9) / repeat);

  reference(Lx, Ly, Threshold, MaxRad, img, h_box, h_norm, h_out);
  verify(size, MaxRad, norm, h_norm, out, h_out, box, h_box);

  std::free(img);
  std::free(norm);
  std::free(h_norm);
  std::free(box);
  std::free(h_box);
  std::free(out);
  std::free(h_out);
  platform.finalize();
  return 0;
}
