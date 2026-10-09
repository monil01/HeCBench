#include <iris/iris.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "../scan2-hip/scan.h"

static void scanLargeArraysCPUReference(float* output, float* input, const unsigned int length) {
  output[0] = 0;
  for (unsigned int i = 1; i < length; ++i) {
    output[i] = input[i - 1] + output[i - 1];
  }
}

static void run_scan(
    const char* label,
    int iterations,
    int length,
    iris::DMem& mem_input,
    iris::DMem& mem_output)
{
  int length_arg = length;
  size_t gws[1] = { 1 };
  size_t lws[1] = { 1 };

  {
    iris::Task task;
    void* params[3] = { &mem_input, &mem_output, &length_arg };
    int pinfo[3] = { iris_r, iris_w, sizeof(length_arg) };
    task.kernel("exclusive_scan_serial", 1, nullptr, gws, lws, 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < iterations; n++) {
    iris::Task task;
    void* params[3] = { &mem_input, &mem_output, &length_arg };
    int pinfo[3] = { iris_r, iris_w, sizeof(length_arg) };
    task.kernel("exclusive_scan_serial", 1, nullptr, gws, lws, 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of IRIS " << label << " exclusive scan: "
            << time * 1e-3f / iterations << " (us)\n";
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <repeat> <input length>\n";
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int iterations = std::atoi(argv[1]);
  int length = std::atoi(argv[2]);

  if (iterations < 1) {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    platform.finalize();
    return -1;
  }
  if (!isPowerOf2(length)) {
    length = roundToPowerOf2(length);
  }

  unsigned int sizeBytes = static_cast<unsigned int>(length) * sizeof(float);
  float* input = static_cast<float*>(std::malloc(sizeBytes));
  float* output = static_cast<float*>(std::malloc(sizeBytes));
  float* verificationOutput = static_cast<float*>(std::malloc(sizeBytes));
  if (!input || !output || !verificationOutput) {
    std::cerr << "host allocation failed\n";
    std::free(input);
    std::free(output);
    std::free(verificationOutput);
    platform.finalize();
    return 1;
  }

  fillRandom<float>(input, length, 1, 0, 255);
  std::memset(output, 0, sizeBytes);
  std::memset(verificationOutput, 0, sizeBytes);
  scanLargeArraysCPUReference(verificationOutput, input, length);

  iris::DMem mem_input(input, sizeBytes);
  iris::DMem mem_output(output, sizeBytes);

  std::cout << "Executing kernel for " << iterations << " iterations\n";
  std::cout << "-------------------------------------------\n";

  run_scan("serial", iterations, length, mem_input, mem_output);
  mem_output.fetch(output);
  if (compare<float>(output, verificationOutput, length, static_cast<float>(0.001))) {
    std::cout << "PASS" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
  }

  run_scan("serial second-pass", iterations, length, mem_input, mem_output);
  mem_output.fetch(output);
  if (compare<float>(output, verificationOutput, length, static_cast<float>(0.001))) {
    std::cout << "PASS" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
  }

  std::free(input);
  std::free(output);
  std::free(verificationOutput);
  platform.finalize();
  return 0;
}
