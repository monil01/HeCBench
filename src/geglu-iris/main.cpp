#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include "reference.h"

#define BLOCK_SIZE 256

static void launch_geglu(iris::DMem& mem_output, iris::DMem& mem_input, int n, int dim_last)
{
  uint64_t total = static_cast<uint64_t>(n) * static_cast<uint64_t>(dim_last);
  size_t gws[1] = { ((static_cast<size_t>(total) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE };
  size_t lws[1] = { BLOCK_SIZE };

  iris::Task task;
  void* params[5] = {
    &mem_output,
    &mem_input,
    &total,
    &n,
    &dim_last
  };
  int pinfo[5] = {
    iris_w,
    iris_r,
    sizeof(total),
    sizeof(n),
    sizeof(dim_last)
  };
  task.kernel("geglu_kernel", 1, nullptr, gws, lws, 5, params, pinfo);
  task.flush_out(mem_output);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::default_random_engine generator(123);
  std::uniform_real_distribution<float> distribution(-6.f, 6.f);

  float *x_and_gate, *output, *output_ref;
  for (int batch = 1; batch <= 4; batch = batch * 4) {
    for (int shape = 128; shape <= 512; shape = shape * 2) {
      for (int dim_last = 1280; dim_last <= 1280; dim_last = dim_last * 2) {
        uint64_t nelems = static_cast<uint64_t>(batch) * shape * dim_last * 2;
        uint64_t nelems_bytes = nelems * sizeof(float);
        x_and_gate = static_cast<float*>(std::malloc(nelems_bytes));
        output_ref = static_cast<float*>(std::malloc(nelems_bytes / 2));
        output = static_cast<float*>(std::malloc(nelems_bytes / 2));

        if (!x_and_gate || !output_ref || !output) {
          std::fprintf(stderr, "host allocation failed\n");
          std::free(x_and_gate);
          std::free(output_ref);
          std::free(output);
          platform.finalize();
          return 1;
        }

        for (uint64_t i = 0; i < nelems; i++) {
          x_and_gate[i] = distribution(generator);
        }
        geglu_reference(output_ref, x_and_gate, batch * shape, dim_last);
        iris::DMem mem_input(x_and_gate, nelems * sizeof(float));
        iris::DMem mem_output(output, (nelems / 2) * sizeof(float));
        launch_geglu(mem_output, mem_input, batch * shape, dim_last);
        mem_output.fetch(output);

        bool ok = true;
        for (uint64_t i = 0; i < nelems / 2; i++) {
          if (std::fabs(output[i] - output_ref[i]) > 1e-3f) {
            ok = false;
            break;
          }
        }

        std::printf("%s\n", ok ? "PASS" : "FAIL");
        std::free(x_and_gate);
        std::free(output);
        std::free(output_ref);
      }
    }
  }

  for (int batch = 1; batch <= 16; batch = batch * 4) {
    for (int shape = 4096; shape <= 8192; shape = shape * 2) {
      for (int dim_last = 1280; dim_last <= 5120; dim_last = dim_last * 2) {
        uint64_t nelems = static_cast<uint64_t>(batch) * shape * dim_last * 2;
        uint64_t nelems_bytes = nelems * sizeof(float);
        x_and_gate = static_cast<float*>(std::malloc(nelems_bytes));
        output = static_cast<float*>(std::malloc(nelems_bytes / 2));

        if (!x_and_gate || !output) {
          std::fprintf(stderr, "host allocation failed\n");
          std::free(x_and_gate);
          std::free(output);
          platform.finalize();
          return 1;
        }

        for (uint64_t i = 0; i < nelems; i++) {
          x_and_gate[i] = distribution(generator);
        }

        iris::DMem mem_input(x_and_gate, nelems * sizeof(float));
        iris::DMem mem_output(output, nelems_bytes / 2);
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; i++) {
          launch_geglu(mem_output, mem_input, batch * shape, dim_last);
        }
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::printf("Batch size: %d, sequence length: %d, hidden dimension: %d\n", batch, shape, dim_last);
        std::printf("Average execution time of GeGLU kernel: %f (us)\n", (time * 1e-3f) / repeat);
        std::free(x_and_gate);
        std::free(output);
      }
    }
  }

  platform.finalize();
  return 0;
}
