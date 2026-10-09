#include <iris/iris.hpp>

#include <omp.h>

#include <cstdio>
#include <cstdlib>

#include "reference.h"

#define GPU_NUM_THREADS 256

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::printf("%s Starting...\n\n", argv[0]);
  const int repeat = std::atoi(argv[1]);

  const int num_gpus = 1;
  std::printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  std::printf("number of devices:\t%d\n", num_gpus);

  const unsigned int nwords = num_gpus * 33554432U;
  const unsigned int nbytes = nwords * sizeof(int);
  const int b = 3;
  int* a = static_cast<int*>(std::malloc(nbytes));

  if (a == nullptr) {
    std::printf("couldn't allocate CPU memory\n");
    platform.finalize();
    return 1;
  }

  double overhead = 0.0;
  int errors = 0;

  for (int i = 0; i < 2; i++) {
    for (int f = 1; f <= 32; f = f * 2) {
      const double start = omp_get_wtime();
      const unsigned int num_cpu_threads = f * num_gpus;

      for (unsigned int cpu_thread_id = 0; cpu_thread_id < num_cpu_threads; cpu_thread_id++) {
        const unsigned int nwords_per_kernel = nwords / num_cpu_threads;
        const unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
        int* sub_a = a + cpu_thread_id * nwords_per_kernel;

        for (unsigned int n = 0; n < nwords_per_kernel; n++) {
          sub_a[n] = n + cpu_thread_id * nwords_per_kernel;
        }

        iris::DMem mem_a(sub_a, nbytes_per_kernel);
        const size_t gws[1] = { static_cast<size_t>(nwords_per_kernel) };
        const size_t lws[1] = { GPU_NUM_THREADS };

        iris::Task task;
        void* params[3] = { &mem_a, const_cast<int*>(&b), const_cast<int*>(&repeat) };
        int pinfo[3] = { iris_rw, sizeof(b), sizeof(repeat) };
        task.kernel("addConstant", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 3, params, pinfo);
        task.flush_out(mem_a);
        task.submit(iris_gpu, nullptr, 1);
      }

      const double end = omp_get_wtime();
      std::printf("Work took %f seconds with %d CPU threads\n", end - start, f * num_gpus);

      if (f == 1) {
        if (i == 0) {
          overhead = end - start;
        } else {
          overhead = overhead - (end - start);
        }
      }

      const bool bResult = correctResult(a, nwords, b, repeat);
      std::printf("%s\n", bResult ? "PASS" : "FAIL");
      if (!bResult) {
        errors++;
      }
    }
  }

  std::printf("Runtime overhead of first run is %f seconds\n", overhead);

  std::free(a);
  platform.finalize();
  return errors == 0 ? 0 : 1;
}
