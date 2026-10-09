#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "reference.h"

static void launch_matern(
    int num_targets,
    float l,
    iris::DMem& mem_sources,
    iris::DMem& mem_targets,
    iris::DMem& mem_weights,
    iris::DMem& mem_result)
{
  const size_t gws[2] = {
    static_cast<size_t>(((num_targets + SX - 1) / SX) * SX),
    static_cast<size_t>(SY)
  };
  const size_t lws[2] = { SX, SY };

  iris::Task task;
  void* params[6] = {
    &num_targets,
    &l,
    &mem_sources,
    &mem_targets,
    &mem_weights,
    &mem_result
  };
  int pinfo[6] = {
    sizeof(num_targets),
    sizeof(l),
    iris_r,
    iris_r,
    iris_r,
    iris_w
  };
  task.kernel("matern_kernel", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 6, params, pinfo);
  task.flush_out(mem_result);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int npoints = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  const int source_size = nsources * 3;
  const int source_size_byte = source_size * static_cast<int>(sizeof(float));
  const int weight_size = nsources;
  const int weight_size_byte = weight_size * static_cast<int>(sizeof(float));
  const int ntargets = npoints * npoints * npoints;
  const int target_size = ntargets * 3;
  const int target_size_byte = target_size * static_cast<int>(sizeof(float));
  const int result_size_byte = ntargets * static_cast<int>(sizeof(float));

  float* sources = static_cast<float*>(std::malloc(source_size_byte));
  float* targets = static_cast<float*>(std::malloc(target_size_byte));
  float* weights = static_cast<float*>(std::malloc(weight_size_byte));
  float* result = static_cast<float*>(std::malloc(result_size_byte));
  float* result_ref = static_cast<float*>(std::malloc(result_size_byte));
  if (!sources || !targets || !weights || !result || !result_ref) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(sources);
    std::free(targets);
    std::free(weights);
    std::free(result);
    std::free(result_ref);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (int i = 0; i < source_size; i++) {
    sources[i] = std::rand() / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < weight_size; i++) {
    weights[i] = std::rand() / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < target_size; i++) {
    targets[i] = std::rand() / static_cast<float>(RAND_MAX);
  }

  iris::DMem mem_sources(sources, source_size_byte);
  iris::DMem mem_weights(weights, weight_size_byte);
  iris::DMem mem_targets(targets, target_size_byte);
  iris::DMem mem_result(result, result_size_byte);

  float l = 0.1f;
  const int ntargets_small = 16 * 16 * 16;
  std::printf("------------------------------------------------------------\n");
  std::printf("Verifying the kernel results with the problem size (16 cube)\n");
  std::printf("------------------------------------------------------------\n");

  int errors = 0;
  while (l <= 1e5f) {
    launch_matern(ntargets_small, l, mem_sources, mem_targets, mem_weights, mem_result);
    matern_kernel_reference(nsources, ntargets_small, l, sources, targets, weights, result_ref);
    mem_result.fetch(result);

    bool ok = true;
    for (int i = 0; i < ntargets_small; i++) {
      if (std::fabs(result[i] - result_ref[i]) > 1e-3f) {
        std::printf("@%d actual=%f expected=%f\n", i, result[i], result_ref[i]);
        ok = false;
        break;
      }
    }
    std::printf("Length scale = %.1e check = %s\n", l, ok ? "PASS" : "FAIL");
    errors += ok ? 0 : 1;
    l = l * 10.f;
  }

  std::printf("--------------------------------------------------------------------\n");
  std::printf("Timing the kernel execution with the problem size (%d cube)\n", npoints);
  std::printf("--------------------------------------------------------------------\n");

  l = 0.1f;
  while (l <= 1e5f) {
    std::printf("Warmup..\n");
    for (int i = 0; i < repeat; i++) {
      launch_matern(ntargets, l, mem_sources, mem_targets, mem_weights, mem_result);
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      launch_matern(ntargets, l, mem_sources, mem_targets, mem_weights, mem_result);
    }
    const auto end = std::chrono::steady_clock::now();
    const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Length scale = %.1e ", l);
    std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
    l = l * 10.f;
  }

  std::free(sources);
  std::free(weights);
  std::free(targets);
  std::free(result);
  std::free(result_ref);
  platform.finalize();
  return errors == 0 ? 0 : 1;
}
