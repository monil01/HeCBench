#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "reference.h"

static bool check_output(const float* got, const float* ref, int n)
{
  for (int i = 0; i < n; i++) {
    if (std::fabs(got[i] - ref[i]) > 1e-3f) {
      std::printf("%f %f\n", got[i], ref[i]);
      return false;
    }
  }
  return true;
}

static bool run_kernel(const char* kernel_name, iris::DMem& mem_src, iris::DMem& mem_dst,
                       float* h_dst, const float* h_src, int n, int w, int h,
                       float focal, int repeat, float* r_dst)
{
  reference<float>(h_src, n, focal, w, h, r_dst);
  const size_t gws[2] = {static_cast<size_t>(((w + 15) / 16) * 16),
                         static_cast<size_t>(((h + 15) / 16) * 16)};
  const size_t lws[2] = {16, 16};
  void* params[6] = {&mem_src, &n, &focal, &w, &h, &mem_dst};
  int pinfo[6] = {iris_r, sizeof(n), sizeof(focal), sizeof(w), sizeof(h), iris_w};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    task.kernel(kernel_name, 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                6, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_dst);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of %s: %f (ms)\n", kernel_name, (time * 1e-6f) / repeat);
  return check_output(h_dst, r_dst, w * h);
}

static bool surfelRenderTest(int n, int w, int h, int repeat)
{
  const int src_size = n * COL_DIM;
  const int dst_size = w * h;
  float* r_dst = static_cast<float*>(std::malloc(dst_size * sizeof(float)));
  float* h_dst = static_cast<float*>(std::malloc(dst_size * sizeof(float)));
  float* h_src = static_cast<float*>(std::malloc(src_size * sizeof(float)));
  if (!r_dst || !h_dst || !h_src) return false;

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis1(-5.0f, 5.0f);
  std::uniform_real_distribution<float> dis2(0.3f, 5.0f);
  std::uniform_real_distribution<float> dis3(-1.0f, 1.0f);
  std::uniform_real_distribution<float> dis4(4e-4f, 2.5e-3f);
  for (int i = 0; i < n; i++) {
    h_src[i * COL_DIM + COL_P_X] = dis1(gen);
    h_src[i * COL_DIM + COL_P_Y] = dis1(gen);
    h_src[i * COL_DIM + COL_P_Z] = dis2(gen);
    float nx = dis3(gen);
    float ny = dis3(gen);
    float nz = dis3(gen);
    float s = std::sqrt(nx * nx + ny * ny + nz * nz);
    h_src[i * COL_DIM + COL_N_X] = nx / s;
    h_src[i * COL_DIM + COL_N_Y] = ny / s;
    h_src[i * COL_DIM + COL_N_Z] = nz / s;
    h_src[i * COL_DIM + COL_RSq] = dis4(gen);
  }

  iris::DMem mem_src(h_src, src_size * sizeof(float));
  iris::DMem mem_dst(h_dst, dst_size * sizeof(float));
  float inverseFocalLength[3] = {0.005f, 0.02f, 0.036f};

  bool ok = true;
  for (int f = 0; f < 3; f++) {
    std::printf("\nf = %d\n", f);
    ok &= run_kernel("surfel_render", mem_src, mem_dst, h_dst, h_src, n, w, h,
                     inverseFocalLength[f], repeat, r_dst);
    if (!ok) break;
    ok &= run_kernel("surfel_render_tile", mem_src, mem_dst, h_dst, h_src, n, w, h,
                     inverseFocalLength[f], repeat, r_dst);
    if (!ok) break;
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(r_dst);
  std::free(h_dst);
  std::free(h_src);
  return ok;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <number of surfels> <output width> <output height> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int n = std::atoi(argv[1]);
  int w = std::atoi(argv[2]);
  int h = std::atoi(argv[3]);
  int repeat = std::atoi(argv[4]);

  std::printf("-------------------------------------\n");
  std::printf(" surfelRenderTest with type float32  \n");
  std::printf("-------------------------------------\n");
  bool ok = surfelRenderTest(n, w, h, repeat);

  platform.finalize();
  return ok ? 0 : 1;
}
