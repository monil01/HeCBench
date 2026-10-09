#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define THREADS_PER_BLOCK 256
#define LAUNCH_ITERATIONS 3

static void init_input(float* a, size_t size)
{
  std::srand(123);
  for (size_t i = 0; i < size; i++) {
    a[i] = (std::rand() & 0xFF) / static_cast<float>(RAND_MAX);
  }
}

static double reference_sum(const std::vector<float>& input)
{
  double result = 0.0;
  for (float value : input) {
    result += static_cast<double>(value);
  }
  return result;
}

static void launch_reduce(iris::DMem& mem_input,
                          iris::DMem& mem_output,
                          size_t input_size,
                          size_t num_blocks)
{
  const size_t gws[1] = { num_blocks * THREADS_PER_BLOCK };
  const size_t lws[1] = { THREADS_PER_BLOCK };
  iris::Task task;
  void* params[4] = { &mem_input, &mem_output, &input_size, &num_blocks };
  int pinfo[4] = { iris_r, iris_w, sizeof(input_size), sizeof(num_blocks) };
  task.kernel("reduce", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_reduce_final(iris::DMem& mem_output,
                                iris::DMem& mem_result,
                                size_t num_blocks)
{
  const size_t gws[1] = { THREADS_PER_BLOCK };
  const size_t lws[1] = { THREADS_PER_BLOCK };
  iris::Task task;
  void* params[3] = { &mem_output, &mem_result, &num_blocks };
  int pinfo[3] = { iris_r, iris_w, sizeof(num_blocks) };
  task.kernel("reduceFinal", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 3, params, pinfo);
  task.flush_out(mem_result);
  task.submit(iris_gpu, nullptr, 1);
}

static void run_iris_reduction(const char* label,
                               const std::vector<float>& input,
                               iris::DMem& mem_input,
                               iris::DMem& mem_output,
                               iris::DMem& mem_result,
                               size_t input_size,
                               size_t num_blocks)
{
  const double result_ref = reference_sum(input);

  for (int iter = 0; iter < LAUNCH_ITERATIONS; iter++) {
    double result_h = 0.0;
    mem_result.update(&result_h);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; i++) {
      launch_reduce(mem_input, mem_output, input_size, num_blocks);
      launch_reduce_final(mem_output, mem_result, num_blocks);
    }
    mem_result.fetch(&result_h);
    auto end = std::chrono::steady_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("%s\n", (std::fabs(result_h - result_ref) < 1e-6) ? "PASS" : "FAIL");
    std::printf("Execution time of using %s: %f (us)\n\n", label, time * 1e-3f);
  }
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const size_t max_blocks = 512;

  for (size_t size = 512; size <= (1u << 27); size *= 512) {
    std::printf("\n-----------------------------\n");
    std::printf("%zu elements\n", size);
    std::printf("Threads per block  = %d\n", THREADS_PER_BLOCK);
    std::printf("Launch iterations = %d\n", LAUNCH_ITERATIONS);

    std::vector<float> input(size);
    std::vector<double> output(max_blocks, 0.0);
    double result = 0.0;

    init_input(input.data(), size);

    iris::DMem mem_input(input.data(), sizeof(float) * input.size());
    iris::DMem mem_output(output.data(), sizeof(double) * output.size());
    iris::DMem mem_result(&result, sizeof(double));

    run_iris_reduction("Graph", input, mem_input, mem_output, mem_result, size, max_blocks);
    run_iris_reduction("Stream", input, mem_input, mem_output, mem_result, size, max_blocks);
  }

  platform.finalize();
  return EXIT_SUCCESS;
}
