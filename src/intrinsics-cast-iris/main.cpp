#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static constexpr int BLOCK_SIZE = 256;

static void run_cast1(int n, int repeat, iris::DMem& mem_input, iris::DMem& mem_output)
{
  const size_t gws[] = { static_cast<size_t>(((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) };
  const size_t lws[] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    void* params[] = { &n, &mem_input, &mem_output };
    int pinfo[] = { sizeof(n), iris_r, iris_w };
    iris::Task task;
    task.kernel("cast1_intrinsics", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 3, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_output);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the cast intrinsics kernel (from FP): %f (us)\n",
              (time * 1e-3f) / repeat);
}

static void run_cast2(int n, int repeat, iris::DMem& mem_input, iris::DMem& mem_output)
{
  const size_t gws[] = { static_cast<size_t>(((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) };
  const size_t lws[] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    void* params[] = { &n, &mem_input, &mem_output };
    int pinfo[] = { sizeof(n), iris_r, iris_w };
    iris::Task task;
    task.kernel("cast2_intrinsics", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 3, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_output);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the cast intrinsics kernel (to FP): %f (us)\n",
              (time * 1e-3f) / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  const size_t input1_size_bytes = static_cast<size_t>(n) * sizeof(double);
  const size_t output1_size_bytes = static_cast<size_t>(n) * sizeof(long long int);
  const size_t input2_size_bytes = static_cast<size_t>(n) * sizeof(long long int);
  const size_t output2_size_bytes = static_cast<size_t>(n) * sizeof(long long int);

  double* input1 = static_cast<double*>(std::malloc(input1_size_bytes));
  long long int* output1 = static_cast<long long int*>(std::malloc(output1_size_bytes));
  long long int* input2 = static_cast<long long int*>(std::malloc(input2_size_bytes));
  long long int* output2 = static_cast<long long int*>(std::malloc(output2_size_bytes));

  if (!input1 || !output1 || !input2 || !output2) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(input1);
    std::free(output1);
    std::free(input2);
    std::free(output2);
    platform.finalize();
    return 1;
  }

  // Preserve the intended 1..n reciprocal sequence without an out-of-bounds write.
  for (int i = 0; i < n; i++) {
    input1[i] = 22.44 / (i + 1);
    input2[i] = 0x403670A3D70A3D71;
  }

  iris::DMem mem_input1(input1, input1_size_bytes);
  iris::DMem mem_output1(output1, output1_size_bytes);
  iris::DMem mem_input2(input2, input2_size_bytes);
  iris::DMem mem_output2(output2, output2_size_bytes);

  run_cast1(n, repeat, mem_input1, mem_output1);
  mem_output1.fetch(output1);

  long long int checksum1 = 0;
  for (int i = 0; i < n; i++) {
    checksum1 = checksum1 ^ output1[i];
  }
  std::printf("Checksum = %llx\n", checksum1);

  run_cast2(n, repeat, mem_input2, mem_output2);
  mem_output2.fetch(output2);

  long long int checksum2 = 0;
  for (int i = 0; i < n; i++) {
    checksum2 ^= output2[i];
  }
  std::printf("Checksum = %llx\n", checksum2);

  if (const char* path = getenv("HECBENCH_DUMP")) {
    FILE* f = fopen(path, "wb");
    if (!f || fwrite(output1, sizeof(*output1), n, f) != (size_t)n ||
        fwrite(output2, sizeof(*output2), n, f) != (size_t)n || fclose(f)) return 2;
  }

  std::free(input1);
  std::free(output1);
  std::free(input2);
  std::free(output2);
  platform.finalize();
  return 0;
}
