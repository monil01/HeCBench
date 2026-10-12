#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define BUF_SIZE 256
#define PATTERN 0xDEADBEEF

static void matrixTransposeCPUReference(float* output, const float* input,
                                        unsigned int numGroups,
                                        unsigned int subGroupSize) {
  for (unsigned i = 0; i < numGroups; ++i) {
    for (unsigned j = 0; j < subGroupSize; j++) {
      output[i * subGroupSize + j] =
          input[i * subGroupSize + subGroupSize - j - 1];
    }
  }
}

static bool verifyBroadcast(const int* out, const int subGroupSize,
                            int pattern = 0) {
  int expected = pattern;
  if (pattern == 0) {
    for (int i = 0; i < subGroupSize; i++) expected += i;
  }

  int errors = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != expected) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
      break;
    }
  }
  std::cout << (errors == 0 ? "PASS\n" : "FAIL\n");
  return errors == 0;
}

static bool verifyTransposeMatrix(const float* transposeMatrix,
                                  const float* cpuTransposeMatrix,
                                  const int total,
                                  const int subGroupSize) {
  int errors = 0;
  const float eps = 1.0E-6f;
  for (int i = 0; i < total; i++) {
    if (std::fabs(transposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ITEM: " << i << " cpu: " << cpuTransposeMatrix[i]
                << " gpu: " << transposeMatrix[i] << "\n";
      errors++;
      break;
    }
  }
  std::cout << (errors == 0 ? "PASS\n" : "FAIL\n");
  return errors == 0;
}

static void* checkedMalloc(size_t bytes) {
  void* ptr = std::malloc(bytes);
  if (!ptr) {
    std::cerr << "host allocation failed\n";
    std::exit(1);
  }
  return ptr;
}

static void launch_broadcast(const char* kernel_name, int arg, int repeat,
                             iris::DMem& mem_out) {
  const int warmup = repeat;
  size_t gws[1] = {BUF_SIZE};
  size_t lws[1] = {BUF_SIZE};

  for (int n = 0; n < warmup + repeat; n++) {
    iris::Task task;
    if (arg == 0) {
      void* params[1] = {&mem_out};
      int pinfo[1] = {iris_w};
      task.kernel(kernel_name, 1, nullptr, gws, lws, 1, params, pinfo);
    } else {
      void* params[2] = {&arg, &mem_out};
      int pinfo[2] = {sizeof(arg), iris_w};
      task.kernel(kernel_name, 1, nullptr, gws, lws, 2, params, pinfo);
    }
    if (n == warmup + repeat - 1) task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
}

static double timed_broadcast(const char* kernel_name, int arg, int repeat,
                              iris::DMem& mem_out) {
  auto begin = std::chrono::steady_clock::now();
  launch_broadcast(kernel_name, arg, repeat, mem_out);
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  return time * 1e-3 / repeat;
}

static void launch_transpose(int total, int subGroupSize, int repeat,
                             iris::DMem& mem_out, iris::DMem& mem_in) {
  const int warmup = repeat;
  size_t gws[1] = {static_cast<size_t>(total)};
  size_t lws[1] = {static_cast<size_t>(subGroupSize)};

  for (int n = 0; n < warmup + repeat; n++) {
    iris::Task task;
    void* params[2] = {&mem_out, &mem_in};
    int pinfo[2] = {iris_w, iris_r};
    task.kernel("transpose_shfl", 1, nullptr, gws, lws, 2, params, pinfo);
    if (n == warmup + repeat - 1) task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
}

static double timed_transpose(int total, int subGroupSize, int repeat,
                              iris::DMem& mem_out, iris::DMem& mem_in) {
  auto begin = std::chrono::steady_clock::now();
  launch_transpose(total, subGroupSize, repeat, mem_out, mem_in);
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  return time * 1e-3 / repeat;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <repeat for broadcast> <repeat for matrix transpose>\n";
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int repeat2 = std::atoi(argv[2]);
  int errors = 0;

  std::cout
      << "Broadcast using the shuffle xor function (subgroup sizes 8, 16, and "
         "32) \n";
  int* out = static_cast<int*>(checkedMalloc(sizeof(int) * BUF_SIZE));
  iris::DMem mem_out(out, sizeof(int) * BUF_SIZE);

  double time_us =
      timed_broadcast("bcast_shfl_xor_sg8", 0, repeat, mem_out);
  std::cout << "Average kernel time (subgroup size = 8): " << time_us
            << " (us)\n";
  mem_out.fetch(out);
  errors += verifyBroadcast(out, 8) ? 0 : 1;

  time_us = timed_broadcast("bcast_shfl_xor_sg16", 0, repeat, mem_out);
  std::cout << "Average kernel time (subgroup size = 16): " << time_us
            << " (us)\n";
  mem_out.fetch(out);
  errors += verifyBroadcast(out, 16) ? 0 : 1;

  time_us = timed_broadcast("bcast_shfl_xor_sg32", 0, repeat, mem_out);
  std::cout << "Average kernel time (subgroup size = 32): " << time_us
            << " (us)\n";
  mem_out.fetch(out);
  errors += verifyBroadcast(out, 32) ? 0 : 1;

  std::cout
      << "Broadcast using the shuffle function (subgroup sizes 8, 16, and 32) "
         "\n";

  time_us = timed_broadcast("bcast_shfl_sg8", PATTERN, repeat, mem_out);
  std::cout << "Average kernel time (subgroup size = 8): " << time_us
            << " (us)\n";
  mem_out.fetch(out);
  errors += verifyBroadcast(out, 8, PATTERN) ? 0 : 1;

  time_us = timed_broadcast("bcast_shfl_sg16", PATTERN, repeat, mem_out);
  std::cout << "Average kernel time (subgroup size = 16): " << time_us
            << " (us)\n";
  mem_out.fetch(out);
  errors += verifyBroadcast(out, 16, PATTERN) ? 0 : 1;

  time_us = timed_broadcast("bcast_shfl_sg32", PATTERN, repeat, mem_out);
  std::cout << "Average kernel time (subgroup size = 32): " << time_us
            << " (us)\n";
  mem_out.fetch(out);
  errors += verifyBroadcast(out, 32, PATTERN) ? 0 : 1;

  std::free(out);

  std::cout << "matrix transpose using the shuffle function (subgroup sizes "
               "are 8, 16, and 32)\n";

  const int total = 1 << 27;
  float* matrix = static_cast<float*>(checkedMalloc(total * sizeof(float)));
  float* transposeMatrix =
      static_cast<float*>(checkedMalloc(total * sizeof(float)));
  float* cpuTransposeMatrix =
      static_cast<float*>(checkedMalloc(total * sizeof(float)));

  for (int i = 0; i < total; i++) {
    matrix[i] = static_cast<float>(i) * 10.0f;
  }

  iris::DMem mem_matrix(matrix, total * sizeof(float));
  iris::DMem mem_transpose(transposeMatrix, total * sizeof(float));

  time_us = timed_transpose(total, 8, repeat2, mem_transpose, mem_matrix);
  std::cout << "Average kernel time (subgroup size = 8): " << time_us
            << " (us)\n";
  mem_transpose.fetch(transposeMatrix);
  matrixTransposeCPUReference(cpuTransposeMatrix, matrix, total / 8, 8);
  errors += verifyTransposeMatrix(transposeMatrix, cpuTransposeMatrix, total, 8)
                ? 0
                : 1;

  time_us = timed_transpose(total, 16, repeat2, mem_transpose, mem_matrix);
  std::cout << "Average kernel time (subgroup size = 16): " << time_us
            << " (us)\n";
  mem_transpose.fetch(transposeMatrix);
  matrixTransposeCPUReference(cpuTransposeMatrix, matrix, total / 16, 16);
  errors +=
      verifyTransposeMatrix(transposeMatrix, cpuTransposeMatrix, total, 16) ? 0
                                                                           : 1;

  time_us = timed_transpose(total, 32, repeat2, mem_transpose, mem_matrix);
  std::cout << "Average kernel time (subgroup size = 32): " << time_us
            << " (us)\n";
  mem_transpose.fetch(transposeMatrix);
  matrixTransposeCPUReference(cpuTransposeMatrix, matrix, total / 32, 32);
  errors +=
      verifyTransposeMatrix(transposeMatrix, cpuTransposeMatrix, total, 32) ? 0
                                                                           : 1;

  std::free(matrix);
  std::free(transposeMatrix);
  std::free(cpuTransposeMatrix);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}
