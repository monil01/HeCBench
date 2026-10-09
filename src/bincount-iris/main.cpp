#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define threadsPerBlock 256

#include "../bincount-cuda/reference.h"

static void launch_zero(iris::DMem& output_mem, int output_size) {
  const size_t gws[] = {static_cast<size_t>(output_size)};
  const size_t lws[] = {threadsPerBlock};
  iris::Task task;
  void* params[] = {&output_mem, &output_size};
  int pinfo[] = {iris_w, sizeof(output_size)};
  task.kernel("zero_int", 1, nullptr, const_cast<size_t*>(gws),
              const_cast<size_t*>(lws), 2, params, pinfo);
  task.flush_out(output_mem);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_bincount(iris::DMem& output_mem, iris::DMem& input_mem,
                            int nbins, float minvalue, float maxvalue,
                            int input_size, int output_size, int repeat) {
  const size_t gws[] = {static_cast<size_t>(input_size)};
  const size_t lws[] = {threadsPerBlock};
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[] = {&output_mem, &input_mem, &nbins, &minvalue, &maxvalue,
                      &input_size, &output_size};
    int pinfo[] = {iris_rw, iris_r, sizeof(nbins),     sizeof(minvalue),
                   sizeof(maxvalue), sizeof(input_size), sizeof(output_size)};
    task.kernel("bincount_global_int_float_int", 1, nullptr,
                const_cast<size_t*>(gws), const_cast<size_t*>(lws), 7, params,
                pinfo);
    task.flush_out(output_mem);
    task.submit(iris_gpu, nullptr, 1);
  }
}

template <typename output_t, typename input_t, typename IndexType>
void eval(IndexType input_size, int repeat) {
  std::vector<input_t> input(input_size);

  std::default_random_engine generator(123);
  std::normal_distribution<input_t> distribution(5.0, 2.0);
  for (IndexType i = 0; i < input_size; i++) input[i] = distribution(generator);

  auto min_iter = std::min_element(input.begin(), input.end());
  auto max_iter = std::max_element(input.begin(), input.end());

  input_t input_minvalue = *min_iter;
  input_t input_maxvalue = *max_iter;
  std::printf("Input min, max values: (%f %f)\n", (float)input_minvalue,
              (float)input_maxvalue);

  int maxSharedMemory = 65536;
  std::printf("Maximum shared local memory size per block in bytes: %d\n",
              maxSharedMemory);

  iris::DMem input_mem(input.data(), input.size() * sizeof(input_t));

  for (IndexType nbins = 768; nbins <= 768 * 32; nbins = nbins * 2) {
    std::printf("\nNumber of bins: %d\n", nbins);
    IndexType sharedMem = nbins * sizeof(output_t);

    IndexType output_size = nbins;
    size_t output_size_bytes = sizeof(output_t) * output_size;
    std::vector<output_t> output(output_size);
    std::vector<output_t> output_r(output_size);

    reference<output_t, input_t, IndexType>(
        output_r.data(), input.data(), nbins, input_minvalue, input_maxvalue,
        input_size, output_size, repeat);

    iris::DMem output_mem(output.data(), output_size_bytes);

    std::printf("bincount using global atomics\n");
    std::fill(output.begin(), output.end(), 0);
    launch_zero(output_mem, output_size);
    auto start = std::chrono::steady_clock::now();
    launch_bincount(output_mem, input_mem, nbins, input_minvalue, input_maxvalue,
                    input_size, output_size, repeat);
    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::printf("Average execution time of bincount kernel: %f (us)\n",
                (time * 1e-3f) / repeat);
    int status = std::memcmp(output.data(), output_r.data(), output_size_bytes);
    std::printf("%s\n", status ? "FAIL" : "PASS");

    if (sharedMem <= maxSharedMemory) {
      std::printf("\n");
      std::printf("bincount using global and local atomics\n");
      std::fill(output.begin(), output.end(), 0);
      launch_zero(output_mem, output_size);
      start = std::chrono::steady_clock::now();
      launch_bincount(output_mem, input_mem, nbins, input_minvalue,
                      input_maxvalue, input_size, output_size, repeat);
      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                 .count();
      std::printf("Average execution time of bincount kernel: %f (us)\n",
                  (time * 1e-3f) / repeat);
      status = std::memcmp(output.data(), output_r.data(), output_size_bytes);
      std::printf("%s\n", status ? "FAIL" : "PASS");
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  eval<int, float, int>(n, repeat);

  platform.finalize();
  return 0;
}
