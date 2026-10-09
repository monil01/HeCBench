#include <iris/iris.hpp>

#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

template <class T>
static T ceil_div(T dividend, T divisor)
{
  return (dividend + divisor - 1) / divisor;
}

static float* make_random_float(size_t n)
{
  float* arr = static_cast<float*>(std::malloc(n * sizeof(float)));
  for (size_t i = 0; i < n; i++) {
    arr[i] = std::rand() / static_cast<float>(RAND_MAX) * 2.f - 1.f;
  }
  return arr;
}

static void permuate_cpu(float* inp, float* q, float* k, float* v, int B, int T, int C, int NH)
{
  size_t i = 0;
  for (int b = 0; b < B; b++) {
    for (int n = 0; n < NH; n++) {
      for (int t = 0; t < T; t++) {
        for (int c = n * (C / NH); c < (n + 1) * (C / NH); c++) {
          q[i] = inp[b * T * 3 * C + t * 3 * C + c];
          k[i] = inp[b * T * 3 * C + t * 3 * C + C + c];
          v[i] = inp[b * T * 3 * C + t * 3 * C + 2 * C + c];
          i++;
        }
      }
    }
  }
}

static bool validate_result(const float* got, const float* ref, const char* name, size_t n, float tolerance)
{
  int nfaults = 0;
  for (size_t i = 0; i < n; i++) {
    float t_eff = tolerance + std::fabs(ref[i]) * FLT_EPSILON;
    if (std::fabs(ref[i] - got[i]) > t_eff) {
      std::printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name, i, ref[i], got[i]);
      if (++nfaults >= 10) return false;
    }
  }
  return nfaults == 0;
}

static void permute(iris::DMem& mem_out, iris::DMem& mem_inp, int B, int T, int C, int NH, int block_size)
{
  int hs = C / NH;
  int total_threads = B * T * C;
  size_t gws[1] = {static_cast<size_t>(ceil_div(total_threads, block_size) * block_size)};
  size_t lws[1] = {static_cast<size_t>(block_size)};
  iris::Task task;
  void* params[6] = {&mem_out, &mem_inp, &B, &T, &NH, &hs};
  int pinfo[6] = {iris_w, iris_r, sizeof(B), sizeof(T), sizeof(NH), sizeof(hs)};
  task.kernel("permute_kernel", 1, nullptr, gws, lws, 6, params, pinfo);
  task.flush_out(mem_out);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int B = std::atoi(argv[1]);
  const int repeat_times = std::atoi(argv[2]);
  const int T = 1024;
  const int C = 768;
  const int NH = 12;
  const size_t S = static_cast<size_t>(B) * T * C;

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  float* inp = make_random_float(S * 3);
  float* out = make_random_float(S * 3);
  float* q = make_random_float(S);
  float* k = make_random_float(S);
  float* v = make_random_float(S);
  if (!inp || !out || !q || !k || !v) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }

  permuate_cpu(inp, q, k, v, B, T, C, NH);

  iris::DMem mem_inp(inp, S * 3 * sizeof(float));
  iris::DMem mem_out(out, S * 3 * sizeof(float));

  int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  bool ok = true;
  for (int block_size : block_sizes) {
    std::printf("Checking block size %d.\n", block_size);
    permute(mem_out, mem_inp, B, T, C, NH, block_size);
    mem_out.fetch(out);
    ok = ok && validate_result(out, q, "q", S, 1e-6f);
    ok = ok && validate_result(out + B * T * C, k, "k", S, 1e-6f);
    ok = ok && validate_result(out + 2 * B * T * C, v, "v", S, 1e-6f);
  }
  if (!ok) {
    std::printf("FAIL\n");
    platform.finalize();
    return 1;
  }
  std::printf("All results match. Starting benchmarks.\n\n");

  for (int block_size : block_sizes) {
    float elapsed_time = 0.f;
    for (int i = 0; i < repeat_times; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      permute(mem_out, mem_inp, B, T, C, NH, block_size);
      auto stop = std::chrono::high_resolution_clock::now();
      elapsed_time += std::chrono::duration<float, std::milli>(stop - start).count();
    }
    std::printf("block_size %4d | time %f ms\n", block_size, elapsed_time / repeat_times);
  }
  std::printf("PASS\n");

  std::free(inp);
  std::free(q);
  std::free(k);
  std::free(v);
  std::free(out);
  platform.finalize();
  return 0;
}
