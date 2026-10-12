#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

struct double2 {
  double x;
  double y;
};

#include "../degrid-cuda/degrid.h"

void init_gcf(PRECISION2* gcf, size_t size)
{
  for (size_t sub_x = 0; sub_x < GCF_GRID; sub_x++) {
    for (size_t sub_y = 0; sub_y < GCF_GRID; sub_y++) {
      for (size_t x = 0; x < size; x++) {
        for (size_t y = 0; y < size; y++) {
          PRECISION tmp = std::sin(6.28 * x / size / GCF_GRID) *
                          std::exp(-(1.0 * x * x + 1.0 * y * y * sub_y) /
                                   size / size / 2);
          gcf[size * size * (sub_x + sub_y * GCF_GRID) + x + y * size].x =
              tmp * std::sin(1.0 * x * sub_x / (y + 1));
          gcf[size * size * (sub_x + sub_y * GCF_GRID) + x + y * size].y =
              tmp * std::cos(1.0 * x * sub_x / (y + 1));
        }
      }
    }
  }
}

void degridCPU(PRECISION2* out, PRECISION2* in, PRECISION2* img, PRECISION2* gcf)
{
  gcf += GCF_DIM * (GCF_DIM + 1) / 2;
  for (size_t n = 0; n < NPOINTS; n++) {
    int sub_x = std::floor(GCF_GRID * (in[n].x - std::floor(in[n].x)));
    int sub_y = std::floor(GCF_GRID * (in[n].y - std::floor(in[n].y)));
    int main_x = std::floor(in[n].x);
    int main_y = std::floor(in[n].y);
    PRECISION sum_r = 0.0;
    PRECISION sum_i = 0.0;
    for (int a = -GCF_DIM / 2; a < GCF_DIM / 2; a++) {
      for (int b = -GCF_DIM / 2; b < GCF_DIM / 2; b++) {
        PRECISION r1 = img[main_x + a + IMG_SIZE * (main_y + b)].x;
        PRECISION i1 = img[main_x + a + IMG_SIZE * (main_y + b)].y;
        PRECISION r2 = gcf[GCF_DIM * GCF_DIM * (GCF_GRID * sub_y + sub_x) +
                           GCF_DIM * b + a].x;
        PRECISION i2 = gcf[GCF_DIM * GCF_DIM * (GCF_GRID * sub_y + sub_x) +
                           GCF_DIM * b + a].y;
        if (main_x + a >= 0 && main_y + b >= 0 &&
            main_x + a < IMG_SIZE && main_y + b < IMG_SIZE) {
          sum_r += r1 * r2 - i1 * i2;
          sum_i += r1 * i2 + r2 * i1;
        }
      }
    }
    out[n].x = sum_r;
    out[n].y = sum_i;
  }
  gcf -= GCF_DIM * (GCF_DIM + 1) / 2;
}

template <class T, class Thalf>
int w_comp_sub(const void* A, const void* B)
{
  Thalf quota, rema, quotb, remb;
  rema = std::modf((*((T*)A)).x, &quota);
  remb = std::modf((*((T*)B)).x, &quotb);
  int sub_xa = (int)(GCF_GRID * rema);
  int sub_xb = (int)(GCF_GRID * remb);
  rema = std::modf((*((T*)A)).y, &quota);
  remb = std::modf((*((T*)B)).y, &quotb);
  int suba = (int)(GCF_GRID * rema) + GCF_GRID * sub_xa;
  int subb = (int)(GCF_GRID * remb) + GCF_GRID * sub_xb;
  if (suba > subb) return 1;
  if (suba < subb) return -1;
  return 0;
}

void degridGPU(PRECISION2* out, PRECISION2* in, PRECISION2* img, PRECISION2* gcf)
{
  img -= IMG_SIZE * GCF_DIM + GCF_DIM;

  const size_t img_count = IMG_SIZE * IMG_SIZE + 2 * IMG_SIZE * GCF_DIM + 2 * GCF_DIM;
  const size_t gcf_count = 64 * GCF_DIM * GCF_DIM;
  iris::DMem mem_img(img, sizeof(PRECISION2) * img_count);
  iris::DMem mem_gcf(gcf, sizeof(PRECISION2) * gcf_count);
  iris::DMem mem_out(out, sizeof(PRECISION2) * NPOINTS);
  iris::DMem mem_in(in, sizeof(PRECISION2) * NPOINTS);

  const size_t npts = NPOINTS;
  const size_t img_dim = IMG_SIZE;
  const size_t gws[2] = { NPOINTS, 8 };
  const size_t lws[2] = { 32, 8 };

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < REPEAT; n++) {
    iris::Task task;
    void* params[] = { &mem_out, &mem_in, const_cast<size_t*>(&npts),
                       &mem_img, const_cast<size_t*>(&img_dim), &mem_gcf };
    int pinfo[] = { iris_w, iris_r, sizeof(npts), iris_r, sizeof(img_dim), iris_r };
    task.kernel("degrid_kernel_double", 2, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 6, params, pinfo);
    if (n + 1 == REPEAT) {
      task.flush_out(mem_out);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / REPEAT << " (s)\n";
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  long img_size = (IMG_SIZE * IMG_SIZE + 2 * IMG_SIZE * GCF_DIM + 2 * GCF_DIM) *
                  sizeof(PRECISION2);
  long io_size = sizeof(PRECISION2) * NPOINTS;

  PRECISION2* out = (PRECISION2*)std::malloc(io_size);
  PRECISION2* in = (PRECISION2*)std::malloc(io_size);
  PRECISION2* img = (PRECISION2*)std::malloc(img_size);
  PRECISION2* gcf = (PRECISION2*)std::malloc(64 * GCF_DIM * GCF_DIM * sizeof(PRECISION2));

  std::cout << "img size in bytes: " << img_size << std::endl;
  std::cout << "out size in bytes: " << io_size << std::endl;

  img += IMG_SIZE * GCF_DIM + GCF_DIM;

  init_gcf(gcf, GCF_DIM);
  std::srand(2541617);
  for (size_t n = 0; n < NPOINTS; n++) {
    in[n].x = ((float)std::rand()) / (float)RAND_MAX * 8000;
    in[n].y = ((float)std::rand()) / (float)RAND_MAX * 8000;
  }
  for (size_t x = 0; x < IMG_SIZE; x++) {
    for (size_t y = 0; y < IMG_SIZE; y++) {
      img[x + IMG_SIZE * y].x =
          std::exp(-((x - 1400.0) * (x - 1400.0) +
                     (y - 3800.0) * (y - 3800.0)) / 8000000.0) + 1.0;
      img[x + IMG_SIZE * y].y = 0.4;
    }
  }
  for (int x = -IMG_SIZE * GCF_DIM - GCF_DIM; x < 0; x++) {
    img[x].x = 0.0;
    img[x].y = 0.0;
  }
  for (int x = 0; x < IMG_SIZE * GCF_DIM + GCF_DIM; x++) {
    img[x + IMG_SIZE * IMG_SIZE].x = 0.0;
    img[x + IMG_SIZE * IMG_SIZE].y = 0.0;
  }

  std::qsort(in, NPOINTS, sizeof(PRECISION2), w_comp_sub<PRECISION2, PRECISION>);

  std::cout << "Computing on GPU..." << std::endl;
  degridGPU(out, in, img, gcf);

  std::cout << "Computing on CPU..." << std::endl;
  PRECISION2* out_cpu = (PRECISION2*)std::malloc(sizeof(PRECISION2) * NPOINTS);
  degridCPU(out_cpu, in, img, gcf);

  std::cout << "Checking results against CPU:" << std::endl;
  PRECISION EPS = (sizeof(PRECISION) == sizeof(double)) ? 1e-7 : 1e-1;
  std::cout << "Error bound: " << EPS << std::endl;

  bool ok = true;
  for (size_t n = 0; n < NPOINTS; n++) {
    if (std::fabs(out[n].x - out_cpu[n].x) > EPS ||
        std::fabs(out[n].y - out_cpu[n].y) > EPS) {
      ok = false;
      std::cout << n << ": F(" << in[n].x << ", " << in[n].y << ") = "
                << out[n].x << ", " << out[n].y << " vs. " << out_cpu[n].x
                << ", " << out_cpu[n].y << std::endl;
      break;
    }
  }

  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  img -= GCF_DIM + IMG_SIZE * GCF_DIM;

  std::free(out);
  std::free(in);
  std::free(img);
  std::free(gcf);
  std::free(out_cpu);

  platform.finalize();
  return ok ? 0 : 1;
}
