#include <iris/iris.hpp>
#include "hip_backend.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define BLACK 0
#define WHITE 255

static int round_up(const int x, const int y)
{
  return (x + y - 1) / y;
}

static double morphology(iris::DMem& img,
                         iris::DMem& tmp,
                         const int width,
                         const int height,
                         const int hsize,
                         const int vsize,
                         const char* horiz_kernel,
                         const char* vert_kernel)
{
  const size_t pixels = static_cast<size_t>(width) * height;
  const size_t clear_gws[1] = {round_up(static_cast<int>(pixels), 256) * 256u};
  const size_t clear_lws[1] = {256};
  const size_t gws_h[2] = {static_cast<size_t>(round_up(width, hsize) * hsize),
                           static_cast<size_t>(height)};
  const size_t lws_h[2] = {static_cast<size_t>(hsize), 1};
  const size_t gws_v[2] = {static_cast<size_t>(width),
                           static_cast<size_t>(round_up(height, vsize) * vsize)};
  const size_t lws_v[2] = {1, static_cast<size_t>(vsize)};

  auto start = std::chrono::steady_clock::now();

  {
    iris::Task task;
    void* params[2] = {&tmp, const_cast<size_t*>(&pixels)};
    int pinfo[2] = {iris_w, sizeof(pixels)};
    task.kernel("clear_u8", 1, nullptr, const_cast<size_t*>(clear_gws),
                const_cast<size_t*>(clear_lws), 2, params, pinfo);
    iris_check(task.submit(hip_device, nullptr, 1), "morphology stage");
  }
  {
    iris::Task task;
    void* params[5] = {&tmp, &img, const_cast<int*>(&width),
                       const_cast<int*>(&height), const_cast<int*>(&hsize)};
    int pinfo[5] = {iris_w, iris_r, sizeof(width), sizeof(height), sizeof(hsize)};
    task.kernel(horiz_kernel, 2, nullptr, const_cast<size_t*>(gws_h),
                const_cast<size_t*>(lws_h), 5, params, pinfo);
    iris_check(task.submit(hip_device, nullptr, 1), "morphology stage");
  }
  {
    iris::Task task;
    void* params[5] = {&img, &tmp, const_cast<int*>(&width),
                       const_cast<int*>(&height), const_cast<int*>(&vsize)};
    int pinfo[5] = {iris_w, iris_r, sizeof(width), sizeof(height), sizeof(vsize)};
    task.kernel(vert_kernel, 2, nullptr, const_cast<size_t*>(gws_v),
                const_cast<size_t*>(lws_v), 5, params, pinfo);
    iris_check(task.submit(hip_device, nullptr, 1), "morphology stage");
  }

  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    std::printf("Usage: %s <kernel width> <kernel height> ", argv[0]);
    std::printf("<image width> <image height> <repeat>\n");
    return 1;
  }

  iris::Platform platform;
  iris_check(platform.init(&argc, &argv, 1), "initialize");
  select_hip_device();

  const int hsize = std::atoi(argv[1]);
  const int vsize = std::atoi(argv[2]);
  const int width = std::atoi(argv[3]);
  const int height = std::atoi(argv[4]);
  const int repeat = std::atoi(argv[5]);

  const size_t mem_size = static_cast<size_t>(width) * height * sizeof(unsigned char);
  unsigned char* src_img = static_cast<unsigned char*>(std::malloc(mem_size));
  unsigned char* tmp_img = static_cast<unsigned char*>(std::calloc(mem_size, 1));
  if (!src_img || !tmp_img) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(src_img);
    std::free(tmp_img);
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      src_img[i * width + j] =
          (i == (height / 2 - 1) && j == (width / 2 - 1)) ? WHITE : BLACK;
    }
  }

  iris::DMem img(src_img, mem_size);
  iris::DMem tmp(tmp_img, mem_size);

  double dilate_time = 0.0;
  double erode_time = 0.0;
  for (int n = 0; n < repeat; n++) {
    dilate_time += morphology(img, tmp, width, height, hsize, vsize,
                              "dilate_horiz", "dilate_vert");
    erode_time += morphology(img, tmp, width, height, hsize, vsize,
                             "erode_horiz", "erode_vert");
  }

  iris::Task flush;
  flush.flush_out(img);
  iris_check(flush.submit(hip_device, nullptr, 1), "output flush");

  std::printf("Average kernel execution time (dilate): %f (s)\n",
              (dilate_time * 1e-9f) / repeat);
  std::printf("Average kernel execution time (erode): %f (s)\n",
              (erode_time * 1e-9f) / repeat);

  int s = 0;
  for (size_t i = 0; i < mem_size; i++) {
    s += src_img[i];
  }
  std::printf("%s\n", s == WHITE ? "PASS" : "FAIL");

  if (const char* path=std::getenv("HECBENCH_VALIDATE_DUMP")) {
    FILE* fp=std::fopen(path,"wb");
    if(!fp) return 2;
    size_t count=std::fwrite(src_img,1,mem_size,fp);
    std::fclose(fp);
    if(count!=mem_size) return 2;
  }
  std::free(src_img);
  std::free(tmp_img);
  platform.finalize();
  return s == WHITE ? 0 : 1;
}
