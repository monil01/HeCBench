#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BLOCK_SIZE 256

static int run_norm2(iris::DMem& mem_a,
                     iris::DMem& mem_partial,
                     iris::DMem& mem_result,
                     int n,
                     int num_blocks)
{
  const size_t gws_partial[1] = {static_cast<size_t>(num_blocks) * BLOCK_SIZE};
  const size_t lws[1] = {BLOCK_SIZE};

  iris::Task partial_task;
  void* partial_params[3] = {&mem_partial, &mem_a, &n};
  int partial_pinfo[3] = {iris_w, iris_r, sizeof(n)};
  partial_task.kernel("norm2_partial", 1, nullptr, const_cast<size_t*>(gws_partial),
                      const_cast<size_t*>(lws), 3, partial_params, partial_pinfo);
  int rc = partial_task.submit(iris_gpu, nullptr, 1);
  if (rc != IRIS_SUCCESS) return rc;

  iris::Task final_task;
  void* final_params[3] = {&mem_result, &mem_partial, &num_blocks};
  int final_pinfo[3] = {iris_w, iris_r, sizeof(num_blocks)};
  final_task.kernel("norm2_final", 1, nullptr, const_cast<size_t*>(lws),
                    const_cast<size_t*>(lws), 3, final_params, final_pinfo);
  final_task.flush_out(mem_result);
  return final_task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = std::max(1, std::atoi(argv[1]));
  bool ok = true;
  std::vector<float> result(repeat, 0.0f);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  for (int n = 512 * 1024; n <= 1024 * 1024 * 512; n *= 2) {
    std::vector<float> a(static_cast<size_t>(n));
    double gold = 0.0;
    for (int i = 0; i < n; i++) {
      a[i] = static_cast<float>((i + 1) % 7);
      gold += a[i] * a[i];
    }
    gold = std::sqrt(gold);

    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<double> partial(static_cast<size_t>(num_blocks), 0.0);
    float one_result = 0.0f;

    iris::DMem mem_a(a.data(), a.size() * sizeof(float));
    iris::DMem mem_partial(partial.data(), partial.size() * sizeof(double));
    iris::DMem mem_result(&one_result, sizeof(one_result));

    auto kstart = std::chrono::steady_clock::now();
    for (int j = 0; j < repeat; j++) {
      const int rc = run_norm2(mem_a, mem_partial, mem_result, n, num_blocks);
      if (rc != IRIS_SUCCESS) {
        std::printf("IRIS Snrm2 failed with code %d\n", rc);
        ok = false;
        break;
      }
      result[j] = one_result;
    }
    auto kend = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    std::printf("#elements = %.2f M: average irisSnrm2 execution time = %f (us), performance = %f (Gop/s)\n",
                n / (1024.f * 1024.f), (ktime * 1e-3f) / repeat,
                1.f * (2 * n + 1) * repeat / ktime);

    for (int j = 0; j < repeat; j++) {
      if (std::fabs(static_cast<float>(gold) - result[j]) > 1e-1f) {
        std::printf("FAIL at iteration %d: gold=%f actual=%f for %d elements\n",
                    j, static_cast<float>(gold), result[j], n);
        ok = false;
        break;
      }
    }
  }

  if (ok) std::printf("PASS\n");
  platform.finalize();
  return ok ? 0 : 1;
}
