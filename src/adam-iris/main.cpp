#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "reference.h"

#define BLOCK_SIZE 256

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <vector size> <number of time steps> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int vector_size = std::atoi(argv[1]);
  const int time_step = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  const size_t size_bytes = static_cast<size_t>(vector_size) * sizeof(float);

  float* m = static_cast<float*>(std::malloc(size_bytes));
  float* v = static_cast<float*>(std::malloc(size_bytes));
  float* g = static_cast<float*>(std::malloc(size_bytes));
  float* p = static_cast<float*>(std::malloc(size_bytes));
  float* r = static_cast<float*>(std::malloc(size_bytes));
  if (!m || !v || !g || !p || !r) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(m);
    std::free(v);
    std::free(g);
    std::free(p);
    std::free(r);
    platform.finalize();
    return 1;
  }

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(0, 1);
  for (int i = 0; i < vector_size; i++) {
    m[i] = dist(gen);
    v[i] = dist(gen);
    g[i] = dist(gen);
    r[i] = p[i] = dist(gen);
  }

  iris::DMem mem_m(m, size_bytes);
  iris::DMem mem_v(v, size_bytes);
  iris::DMem mem_g(g, size_bytes);
  iris::DMem mem_p(p, size_bytes);

  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;
  const int mode = ADAM_MODE_0;

  const size_t blocks = (static_cast<size_t>(vector_size) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = {blocks * BLOCK_SIZE};
  const size_t lws[1] = {BLOCK_SIZE};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[13] = {
        &mem_p,
        &mem_m,
        &mem_v,
        &mem_g,
        const_cast<float*>(&beta1),
        const_cast<float*>(&beta2),
        const_cast<float*>(&eps),
        const_cast<float*>(&grad_scale),
        const_cast<float*>(&step_size),
        const_cast<int*>(&time_step),
        const_cast<int*>(&vector_size),
        const_cast<int*>(&mode),
        const_cast<float*>(&decay)};
    int pinfo[13] = {
        iris_rw,
        iris_rw,
        iris_rw,
        iris_r,
        sizeof(beta1),
        sizeof(beta2),
        sizeof(eps),
        sizeof(grad_scale),
        sizeof(step_size),
        sizeof(time_step),
        sizeof(vector_size),
        sizeof(mode),
        sizeof(decay)};
    task.kernel("adam_kernel", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 13, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  iris::Task flush;
  flush.flush_out(mem_p);
  flush.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  mem_p.fetch(p);

  reference<float, float>(
      repeat,
      r,
      m,
      v,
      g,
      beta1,
      beta2,
      eps,
      grad_scale,
      step_size,
      time_step,
      static_cast<size_t>(vector_size),
      static_cast<adamMode_t>(mode),
      decay);

  bool ok = true;
  double cr = 0, cp = 0;
  for (int i = 0; i < vector_size; i++) {
    if (std::fabs(r[i] - p[i]) > 1e-3f) {
      ok = false;
      break;
    }
    cr += r[i];
    cp += p[i];
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  std::printf("Checksum: %lf %lf\n", cr / vector_size, cp / vector_size);

  std::free(p);
  std::free(m);
  std::free(v);
  std::free(g);
  std::free(r);
  platform.finalize();
  return ok ? 0 : 1;
}
