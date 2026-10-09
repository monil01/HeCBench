#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) (((size_t)(x) + ((size) - 1)) & (~((size) - 1)))

static int eval(bool warmup, bool generic_memory, int repeat)
{
  int failures = 0;

  if (generic_memory) {
    std::printf("> Using Generic System Paged Memory (malloc)\n");
  } else {
    std::printf("> Using Host Allocated (IRIS dmem host buffer)\n");
  }

  if (warmup) std::printf("Warmup...\n");

  for (int nelem = 1024 * 1024; nelem <= (1024 * 1024 * 64); nelem *= 2) {
    if (!warmup) std::printf("\nvector length = %d\n", nelem);

    const size_t bytes = static_cast<size_t>(nelem) * sizeof(float);
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    float* a_unaligned = nullptr;
    float* b_unaligned = nullptr;
    float* c_unaligned = nullptr;

    auto start = std::chrono::steady_clock::now();
    if (generic_memory) {
      a_unaligned = static_cast<float*>(std::malloc(bytes + MEMORY_ALIGNMENT));
      b_unaligned = static_cast<float*>(std::malloc(bytes + MEMORY_ALIGNMENT));
      c_unaligned = static_cast<float*>(std::malloc(bytes + MEMORY_ALIGNMENT));
      a = reinterpret_cast<float*>(ALIGN_UP(a_unaligned, MEMORY_ALIGNMENT));
      b = reinterpret_cast<float*>(ALIGN_UP(b_unaligned, MEMORY_ALIGNMENT));
      c = reinterpret_cast<float*>(ALIGN_UP(c_unaligned, MEMORY_ALIGNMENT));
    } else {
      a = static_cast<float*>(std::aligned_alloc(MEMORY_ALIGNMENT, bytes));
      b = static_cast<float*>(std::aligned_alloc(MEMORY_ALIGNMENT, bytes));
      c = static_cast<float*>(std::aligned_alloc(MEMORY_ALIGNMENT, bytes));
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (!warmup) {
      std::printf("Memory allocation (%s): %lf ms\n",
                  generic_memory ? "malloc aligned for IRIS dmem" : "aligned_alloc for IRIS dmem",
                  time * 1e-6);
    }

    if (a == nullptr || b == nullptr || c == nullptr) {
      std::fprintf(stderr, "allocation failed\n");
      return 1;
    }

    for (int n = 0; n < nelem; n++) {
      a[n] = std::rand() / static_cast<float>(RAND_MAX);
      b[n] = std::rand() / static_cast<float>(RAND_MAX);
      c[n] = 0.0f;
    }

    {
      start = std::chrono::steady_clock::now();
      iris::DMem d_a(a, bytes);
      iris::DMem d_b(b, bytes);
      iris::DMem d_c(c, bytes);
      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      if (!warmup) std::printf("IRIS dmem setup: %lf ms\n", time * 1e-6);

      const int threads = 256;
      const int grids = (nelem + threads - 1) / threads;
      const size_t gws[1] = {static_cast<size_t>(grids * threads)};
      const size_t lws[1] = {static_cast<size_t>(threads)};

      start = std::chrono::steady_clock::now();
      for (int n = 0; n < repeat; n++) {
        iris::Task task;
        void* params[4] = {&d_a, &d_b, &d_c, &nelem};
        int pinfo[4] = {iris_r, iris_r, iris_w, sizeof(nelem)};
        task.kernel("vectorAddGPU", 1, nullptr, const_cast<size_t*>(gws),
                    const_cast<size_t*>(lws), 4, params, pinfo);
        task.flush_out(d_c);
        task.submit(iris_gpu, nullptr, 1);
      }
      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      if (!warmup) std::printf("Average kernel execution time: %lf ms\n", time * 1e-6 / repeat);

      d_c.fetch(c);
    }

    if (warmup) {
      float error_norm = 0.0f;
      float ref_norm = 0.0f;

      for (int n = 0; n < nelem; n++) {
        float ref = a[n] + b[n];
        float diff = c[n] - ref;
        error_norm += diff * diff;
        ref_norm += ref * ref;
      }

      error_norm = std::sqrt(error_norm);
      ref_norm = std::sqrt(ref_norm);

      bool ok = error_norm / ref_norm < 1.e-6f;
      std::printf("%s\n", ok ? "SUCCESS" : "FAILURE");
      if (!ok) failures++;
    }

    start = std::chrono::steady_clock::now();
    if (generic_memory) {
      std::free(a_unaligned);
      std::free(b_unaligned);
      std::free(c_unaligned);
    } else {
      std::free(a);
      std::free(b);
      std::free(c);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (!warmup) {
      std::printf("Memory deallocation (%s): %lf ms\n",
                  generic_memory ? "free generic IRIS dmem buffers" : "free aligned IRIS dmem buffers",
                  time * 1e-6);
    }
  }

  if (warmup) std::printf("Done.\n");
  return failures;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  int failures = 0;

  failures += eval(true, false, repeat);
  failures += eval(false, false, repeat);

  failures += eval(true, true, repeat);
  failures += eval(false, true, repeat);

  platform.finalize();
  return failures == 0 ? 0 : 1;
}
