#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void run_kernel(const char* name, int64_t nrow, int64_t ncol,
                       iris::DMem& d_r, iris::DMem& d_m, iris::DMem& d_v)
{
  size_t gws[1] = {static_cast<size_t>(((nrow + 255) / 256) * 256)};
  size_t lws[1] = {256};
  void* params[5] = {&nrow, &ncol, &d_r, &d_m, &d_v};
  int pinfo[5] = {sizeof(nrow), sizeof(ncol), iris_w, iris_r, iris_r};
  iris::Task task;
  task.kernel(name, 1, nullptr, gws, lws, 5, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int64_t nrow = atol(argv[1]);
  const int64_t ncol = atol(argv[2]);
  const int repeat = atoi(argv[3]);

  const int64_t numel = nrow * ncol;
  const int64_t m_bytes = numel * sizeof(float);
  const int64_t v_bytes = ncol * sizeof(float);
  const int64_t r_bytes = nrow * sizeof(float);

  float* m = (float*)malloc(m_bytes);
  float* v = (float*)malloc(v_bytes);
  float* r = (float*)malloc(r_bytes);
  float* r_ref = (float*)malloc(r_bytes);

  srand(123);
  for (int64_t i = 0; i < numel; i++) {
    m[i] = rand() / (float)RAND_MAX;
  }

  for (int64_t i = 0; i < ncol; i++) {
    v[i] = rand() / (float)RAND_MAX;
  }

  for (int64_t i = 0; i < nrow; i++) {
    float val = 0.f;
    for (int64_t j = 0; j < ncol; j++) {
      val += m[i * ncol + j] * v[j];
    }
    r_ref[i] = val;
  }

  iris::DMem d_m(m, m_bytes);
  iris::DMem d_v(v, v_bytes);
  iris::DMem d_r(r, r_bytes);

  printf("Warmup..\n");
  for (int i = 0; i < repeat; i++) {
    run_kernel("tensor_packed_accessor_kernel", nrow, ncol, d_r, d_m, d_v);
    run_kernel("raw_accessor_kernel", nrow, ncol, d_r, d_m, d_v);
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    run_kernel("raw_accessor_kernel", nrow, ncol, d_r, d_m, d_v);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of raw_accessor_kernel: %f (us)\n",
         time * 1e-3f / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    run_kernel("tensor_packed_accessor_kernel", nrow, ncol, d_r, d_m, d_v);
  }

  iris::Task flush;
  flush.flush_out(d_r);
  flush.submit(iris_gpu, nullptr, 1);
  d_r.fetch(r);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of tensor_packed_accessor_kernel: %f (us)\n",
         time * 1e-3f / repeat);

  bool ok = true;
  for (int64_t i = 0; i < nrow; i++) {
    if (fabsf(r[i] - r_ref[i]) > 1e-3f) {
      printf("%f %f\n", r[i], r_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(m);
  free(v);
  free(r);
  free(r_ref);
  platform.finalize();
  return 0;
}
