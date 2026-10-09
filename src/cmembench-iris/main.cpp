#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define VECTOR_SIZE 1024
#define BLOCK_SIZE 256

static void launch_case(const char* name, iris::DMem& mem_input,
                        iris::DMem& mem_output, int repeat, bool flush)
{
  const long gridsize = 4096L * VECTOR_SIZE;
  const size_t gws[1] = {static_cast<size_t>(gridsize)};
  const size_t lws[1] = {BLOCK_SIZE};
  iris::Task task;
  void* params[3] = {&mem_output, &mem_input, &repeat};
  int pinfo[3] = {iris_w, iris_r, sizeof(repeat)};
  task.kernel(name, 1, nullptr, const_cast<size_t*>(gws),
              const_cast<size_t*>(lws), 3, params, pinfo);
  if (flush) task.flush_out(mem_output);
  task.submit(iris_gpu, nullptr, 1);
}

static void test_case(const char* name, size_t width, iris::DMem& mem_input,
                      iris::DMem& mem_output, int* output, int repeat)
{
  for (int i = 0; i < repeat; i++) {
    launch_case(name, mem_input, mem_output, repeat, false);
  }

  *output = 0;
  mem_output.update(output);
  iris_synchronize();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_case(name, mem_input, mem_output, repeat, i == repeat - 1);
  }
  iris_synchronize();
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::printf("%s\t", (*output == VECTOR_SIZE) ? "PASS" : "FAIL");
  std::printf("Average kernel execution time (memory access width = %zu bytes): %f ms\n",
              width, time * 1e-6f / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Constant memory bandwidth microbenchmark\n");
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  const unsigned int datasize = VECTOR_SIZE * sizeof(int);

  int* v = static_cast<int*>(std::malloc(datasize));
  int* output = static_cast<int*>(std::malloc(sizeof(int)));
  if (v == nullptr || output == nullptr) {
    std::fprintf(stderr, "Host allocation failed\n");
    std::free(v);
    std::free(output);
    return 1;
  }
  for (int i = 0; i < VECTOR_SIZE; i++) v[i] = 1;
  *output = 0;

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem mem_input(v, datasize);
  iris::DMem mem_output(output, sizeof(int));
  mem_input.update(v);

  test_case("benchmark_constant_i32", sizeof(int), mem_input, mem_output,
            output, repeat);
  test_case("benchmark_constant_i32x2", sizeof(int) * 2, mem_input, mem_output,
            output, repeat);
  test_case("benchmark_constant_i32x4", sizeof(int) * 4, mem_input, mem_output,
            output, repeat);

  platform.finalize();
  std::free(v);
  std::free(output);
  return 0;
}
