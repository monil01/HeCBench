#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "reference.h"

#define BLOCK_SIZE 256

static void submit_kernel(
    const char* name,
    size_t imgSize,
    iris::DMem& a,
    iris::DMem& b,
    iris::DMem& c,
    iris::DMem& d,
    iris::DMem& e)
{
  const size_t gws[1] = { ((imgSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  iris::Task task;
  void* params[6] = { &imgSize, &a, &b, &c, &d, &e };
  int pinfo[6] = { sizeof(imgSize), iris_r, iris_r, iris_r, iris_r, iris_w };
  task.kernel(name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 6, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void submit_update_background(size_t imgSize, iris::DMem& Img, iris::DMem& Mp, iris::DMem& Bn)
{
  const size_t gws[1] = { ((imgSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  iris::Task task;
  void* params[4] = { &imgSize, &Img, &Mp, &Bn };
  int pinfo[4] = { sizeof(imgSize), iris_r, iris_r, iris_rw };
  task.kernel("updateBackground", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void submit_update_threshold(
    size_t imgSize,
    iris::DMem& Img,
    iris::DMem& Mp,
    iris::DMem& Bn,
    iris::DMem& Tn)
{
  const size_t gws[1] = { ((imgSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  iris::Task task;
  void* params[5] = { &imgSize, &Img, &Mp, &Bn, &Tn };
  int pinfo[5] = { sizeof(imgSize), iris_r, iris_r, iris_r, iris_rw };
  task.kernel("updateThreshold", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void submit_merge(size_t imgSize, iris::DMem& Img, iris::DMem& Img1, iris::DMem& Img2, iris::DMem& Tn, iris::DMem& Bn)
{
  const size_t gws[1] = { ((imgSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  iris::Task task;
  void* params[6] = { &imgSize, &Img, &Img1, &Img2, &Tn, &Bn };
  int pinfo[6] = { sizeof(imgSize), iris_r, iris_r, iris_r, iris_rw, iris_rw };
  task.kernel("merge", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 6, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <image width> <image height> <merge> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int width = std::atoi(argv[1]);
  const int height = std::atoi(argv[2]);
  const int merged = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  const size_t imgSize = static_cast<size_t>(width) * height;
  const size_t imgSize_bytes = imgSize * sizeof(unsigned char);

  unsigned char* Img = static_cast<unsigned char*>(std::malloc(imgSize_bytes));
  unsigned char* Img1 = static_cast<unsigned char*>(std::malloc(imgSize_bytes));
  unsigned char* Img2 = static_cast<unsigned char*>(std::malloc(imgSize_bytes));
  unsigned char* Bn = static_cast<unsigned char*>(std::malloc(imgSize_bytes));
  unsigned char* Bn_ref = static_cast<unsigned char*>(std::malloc(imgSize_bytes));
  unsigned char* Tn = static_cast<unsigned char*>(std::malloc(imgSize_bytes));
  unsigned char* Tn_ref = static_cast<unsigned char*>(std::malloc(imgSize_bytes));
  unsigned char* Mp = static_cast<unsigned char*>(std::malloc(imgSize_bytes));
  if (!Img || !Img1 || !Img2 || !Bn || !Bn_ref || !Tn || !Tn_ref || !Mp) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }

  std::mt19937 generator(123);
  std::uniform_int_distribution<int> distribute(0, 255);
  for (size_t j = 0; j < imgSize; j++) {
    Bn_ref[j] = Bn[j] = distribute(generator);
    Tn_ref[j] = Tn[j] = 128;
  }

  iris::DMem mem_img_a(Img, imgSize_bytes);
  iris::DMem mem_img_b(Img1, imgSize_bytes);
  iris::DMem mem_img_c(Img2, imgSize_bytes);
  iris::DMem mem_bn(Bn, imgSize_bytes);
  iris::DMem mem_mp(Mp, imgSize_bytes);
  iris::DMem mem_tn(Tn, imgSize_bytes);

  iris::DMem* mem_Img = &mem_img_a;
  iris::DMem* mem_Img1 = &mem_img_b;
  iris::DMem* mem_Img2 = &mem_img_c;

  long time = 0;
  for (int i = 0; i < repeat; i++) {
    for (size_t j = 0; j < imgSize; j++) {
      Img[j] = distribute(generator);
    }
    mem_Img->update(Img);

    iris::DMem* mt = mem_Img2;
    mem_Img2 = mem_Img1;
    mem_Img1 = mem_Img;
    mem_Img = mt;

    unsigned char* t = Img2;
    Img2 = Img1;
    Img1 = Img;
    Img = t;

    if (i >= 2) {
      const auto start = std::chrono::steady_clock::now();
      if (merged) {
        submit_merge(imgSize, *mem_Img, *mem_Img1, *mem_Img2, mem_tn, mem_bn);
      } else {
        submit_kernel("findMovingPixels", imgSize, *mem_Img, *mem_Img1, *mem_Img2, mem_tn, mem_mp);
        submit_update_background(imgSize, *mem_Img, mem_mp, mem_bn);
        submit_update_threshold(imgSize, *mem_Img, mem_mp, mem_bn, mem_tn);
      }
      const auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      merge_ref(imgSize, Img, Img1, Img2, Tn_ref, Bn_ref);
    }
  }

  const float kernel_time = (repeat <= 2) ? 0 : (time * 1e-3f) / (repeat - 2);
  std::printf("Average kernel execution time: %f (us)\n", kernel_time);

  mem_tn.fetch(Tn);
  mem_bn.fetch(Bn);

  int max_error = 0;
  for (size_t i = 0; i < imgSize; i++) {
    const int err = std::abs(Tn[i] - Tn_ref[i]);
    if (err > max_error) max_error = err;
  }
  for (size_t i = 0; i < imgSize; i++) {
    const int err = std::abs(Bn[i] - Bn_ref[i]);
    if (err > max_error) max_error = err;
  }
  std::printf("Max error is %d\n", max_error);
  std::printf("%s\n", max_error ? "FAIL" : "PASS");

  std::free(Img);
  std::free(Img1);
  std::free(Img2);
  std::free(Tn);
  std::free(Bn);
  std::free(Tn_ref);
  std::free(Bn_ref);
  std::free(Mp);
  platform.finalize();
  return max_error ? 1 : 0;
}
