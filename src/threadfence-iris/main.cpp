#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static constexpr int kBlockSize = 256;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("Usage: %s <repeat> <array length>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);

  const int blocks = kBlockSize;
  const int grids = (N + blocks - 1) / blocks;

  auto* h_array = static_cast<float*>(std::malloc(static_cast<size_t>(N) * sizeof(float)));
  auto* h_result = static_cast<float*>(std::calloc(static_cast<size_t>(grids), sizeof(float)));
  if (!h_array || !h_result) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_array);
    std::free(h_result);
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < N; i++) {
    h_array[i] = -1.f;
  }
  unsigned int h_count = 0;

  iris::DMem mem_array(h_array, static_cast<size_t>(N) * sizeof(float));
  iris::DMem mem_result(h_result, static_cast<size_t>(grids) * sizeof(float));
  iris::DMem mem_count(&h_count, sizeof(unsigned int));

  int n_arg = N;
  double time = 0.0;

  size_t gws[1] = { static_cast<size_t>(grids) * kBlockSize };
  size_t lws[1] = { kBlockSize };

  for (int n = 0; n < repeat; n++) {
    auto start = std::chrono::steady_clock::now();

    iris::Task task;
    void* params[4] = { &mem_array, &n_arg, &mem_count, &mem_result };
    int pinfo[4] = { iris_r, sizeof(n_arg), iris_rw, iris_rw };
    task.kernel("sum", 1, nullptr, gws, lws, 4, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  mem_result.fetch(h_result);
  const float h_sum = h_result[0];
  const bool ok = (h_sum == -1.f * N);

  if (ok) {
    std::printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
  }

  std::free(h_array);
  std::free(h_result);
  platform.finalize();

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
