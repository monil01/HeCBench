#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define VECTOR_SIZE (8 * 1024 * 1024)
#define granularity (8)

static void mixbenchGPU(long size, int compute_iterations, int repeat)
{
  const char* benchtype = "compute with global memory (block strided)";
  std::printf("Trade-off type:%s\n", benchtype);

  std::vector<float> cd(static_cast<size_t>(size), 0.0f);

  const long reduced_grid_size = size / granularity / 128;
  const int block_dim = 256;
  const int grid_dim = static_cast<int>(reduced_grid_size / block_dim);
  size_t gws[1] = { static_cast<size_t>(grid_dim * block_dim) };
  size_t lws[1] = { static_cast<size_t>(block_dim) };

  iris::DMem mem_cd(cd.data(), cd.size() * sizeof(float));
  mem_cd.update(cd.data());

  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = { &mem_cd, &compute_iterations };
    int pinfo[2] = { iris_rw, sizeof(compute_iterations) };
    task.kernel("benchmark_func", 1, nullptr, gws, lws, 2, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = { &mem_cd, &compute_iterations };
    int pinfo[2] = { iris_rw, sizeof(compute_iterations) };
    task.kernel("benchmark_func", 1, nullptr, gws, lws, 2, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_cd);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();

  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  mem_cd.fetch(cd.data());

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (cd[static_cast<size_t>(i)] != 0.0f) {
      if (std::fabs(cd[static_cast<size_t>(i)] - 0.050807f) > 1e-6f) {
        ok = false;
        std::printf("Verification failed at index %d: %f\n", i, cd[static_cast<size_t>(i)]);
        break;
      }
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <compute iterations> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int compute_iterations = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  const unsigned int datasize = VECTOR_SIZE * sizeof(float);

  std::printf("Buffer size: %dMB\n", datasize / (1024 * 1024));
  mixbenchGPU(VECTOR_SIZE, compute_iterations, repeat);

  platform.finalize();
  return 0;
}
