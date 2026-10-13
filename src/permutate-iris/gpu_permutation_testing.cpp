#include "header.h"
#include <iris/iris.hpp>
#include <chrono>
#include <memory>
#include <stdexcept>
#include "validation.h"

static void checked(int rc) {
  if (rc != IRIS_SUCCESS) throw std::runtime_error("IRIS permutation task failed");
}

bool gpu_permutation_testing(double *gpu_runtime, uint32_t *counts, double *results,
                            double mean, double median, uint8_t *data, uint32_t size,
                            uint32_t len, uint32_t N, uint32_t num_block, uint32_t num_thread) {
  ValidationGPULease gpu_lease;
  iris::Platform platform;
  checked(platform.init(nullptr, nullptr, 1));
  bool hip = false;
  for (int d = 0; d < platform.ndevs(); ++d) {
    int backend = 0, type = 0; char name[256] = {}; size_t bytes = 0;
    checked(iris_device_info(d, iris_backend, &backend, &bytes));
    checked(iris_device_info(d, iris_type, &type, &bytes));
    checked(iris_device_info(d, iris_name, name, &bytes));
    printf("DEVICE backend=%d type=%d name=%s\n", backend, type, name);
    hip |= backend == 3 && type == 256;
  }
  if (!hip) throw std::runtime_error("HIP GPU required; CPU fallback forbidden");
  uint32_t loop = 10000 / N + (10000 % N != 0);
  uint32_t blen = size == 1 ? len / 8 + (len % 8 != 0) : 0;
  size_t Nlen = size_t(N) * len, Nblen = size_t(N) * blen;
  uint8_t num_runtest = 0;
  {
    std::unique_ptr<uint8_t[]> shuffled(new uint8_t[Nlen]);
    std::unique_ptr<uint8_t[]> binary(size == 1 ? new uint8_t[Nblen] : nullptr);
    iris::DMem input(data, len), output(shuffled.get(), Nlen);
    iris::DMem original(results, 18 * sizeof(double)), counters(counts, 54 * sizeof(uint32_t));
    std::unique_ptr<iris::DMem> packed;
    if (size == 1) packed.reset(new iris::DMem(binary.get(), Nblen));
    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < loop; ++i) {
      size_t lws[] = {num_thread};
      size_t shuffle_gws[] = {size_t(num_block) * num_thread};
      size_t tests_gws[] = {size_t(num_block) * num_thread * (size == 1 ? 4 : 2)};
      iris::Task shuffle, tests;
      if (size == 1) {
        void* args[] = {&output, packed.get(), &input, &len, &blen, &N};
        int info[] = {iris_w, iris_w, iris_r, sizeof(len), sizeof(blen), sizeof(N)};
        checked(shuffle.kernel("binary_shuffling_kernel", 1, nullptr, shuffle_gws, lws, 6, args, info));
        void* stat[] = {&counters, &original, &mean, &median, &output, packed.get(), &size, &len, &blen, &N, &num_block};
        int sinfo[] = {iris_rw, iris_r, sizeof(mean), sizeof(median), iris_r, iris_r, sizeof(size), sizeof(len), sizeof(blen), sizeof(N), sizeof(num_block)};
        checked(tests.kernel("binary_statistical_tests_kernel", 1, nullptr, tests_gws, lws, 11, stat, sinfo));
      } else {
        void* args[] = {&output, &input, &len, &N};
        int info[] = {iris_w, iris_r, sizeof(len), sizeof(N)};
        checked(shuffle.kernel("shuffling_kernel", 1, nullptr, shuffle_gws, lws, 4, args, info));
        void* stat[] = {&counters, &original, &mean, &median, &output, &size, &len, &N, &num_block};
        int sinfo[] = {iris_rw, iris_r, sizeof(mean), sizeof(median), iris_r, sizeof(size), sizeof(len), sizeof(N), sizeof(num_block)};
        checked(tests.kernel("statistical_tests_kernel", 1, nullptr, tests_gws, lws, 9, stat, sinfo));
      }
      checked(shuffle.submit(iris_gpu, nullptr, 1));
      checked(tests.flush_out(counters));
      if (getenv("PERMUTATE_SHUFFLE_DUMP")) {
        checked(tests.flush_out(output));
        if (packed) checked(tests.flush_out(*packed));
      }
      checked(tests.submit(iris_gpu, nullptr, 1));
      validation_batch(i, counts);
      num_runtest = 0;
      for (int t = 0; t < 18; ++t)
        if ((counts[3*t] + counts[3*t+1] > 5) && (counts[3*t+1] + counts[3*t+2] > 5)) ++num_runtest;
      if (num_runtest == 18 || i + 1 == loop) {
        validation_shuffle(shuffled.get(), Nlen, binary.get(), Nblen);
        break;
      }
    }
    *gpu_runtime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
  int errors = platform.error_count();
  checked(platform.finalize());
  if (errors) throw std::runtime_error("IRIS reported backend errors");
  return num_runtest == 18;
}
