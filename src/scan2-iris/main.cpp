#include <iris/iris.hpp>

#include "scan.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

static void scanLargeArraysCPUReference(float* output, float* input, unsigned int length)
{
  output[0] = 0;
  for (unsigned int i = 1; i < length; ++i) output[i] = input[i - 1] + output[i - 1];
}

static void bScan(unsigned int blockSize, unsigned int len, iris::DMem& inputBuffer,
                  iris::DMem& outputBuffer, iris::DMem& blockSumBuffer)
{
  size_t gws[1] = {static_cast<size_t>(len / 2)};
  size_t lws[1] = {static_cast<size_t>(blockSize / 2)};
  iris::Task task;
  void* params[4] = {&outputBuffer, &inputBuffer, &blockSize, &blockSumBuffer};
  int pinfo[4] = {iris_w, iris_r, sizeof(blockSize), iris_w};
  task.kernel("ScanLargeArrays", 1, nullptr, gws, lws, 4, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void pScan(unsigned int blockSize, unsigned int len, iris::DMem& inputBuffer,
                  iris::DMem& outputBuffer)
{
  size_t gws[1] = {static_cast<size_t>(len / 2)};
  size_t lws[1] = {static_cast<size_t>(len / 2)};
  iris::Task task;
  void* params[3] = {&outputBuffer, &inputBuffer, &blockSize};
  int pinfo[3] = {iris_w, iris_r, sizeof(blockSize)};
  task.kernel("prefixSum", 1, nullptr, gws, lws, 3, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void bAddition(unsigned int blockSize, unsigned int len, iris::DMem& inputBuffer,
                      iris::DMem& outputBuffer)
{
  size_t gws[1] = {static_cast<size_t>(len)};
  size_t lws[1] = {static_cast<size_t>(blockSize)};
  iris::Task task;
  void* params[2] = {&inputBuffer, &outputBuffer};
  int pinfo[2] = {iris_r, iris_rw};
  task.kernel("blockAddition", 1, nullptr, gws, lws, 2, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <repeat> <input length> <block size>\n";
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int iterations = std::atoi(argv[1]);
  int length = std::atoi(argv[2]);
  int blockSizeArg = std::atoi(argv[3]);
  if (iterations < 1) {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    platform.finalize();
    return -1;
  }
  if (!isPowerOf2(length)) length = roundToPowerOf2(length);
  if ((length / blockSizeArg > GROUP_SIZE) && (((length) & (length - 1)) != 0)) {
    std::cout << "Invalid length: " << length << std::endl;
    platform.finalize();
    return -1;
  }

  unsigned int blockSize =
      static_cast<unsigned int>((blockSizeArg < length / 2) ? blockSizeArg : length / 2);
  unsigned int sizeBytes = static_cast<unsigned int>(length * sizeof(float));
  float* input = static_cast<float*>(std::malloc(sizeBytes));
  float* output = static_cast<float*>(std::malloc(sizeBytes));
  fillRandom<float>(input, length, 1, 0, 255);

  float t = std::log(static_cast<float>(length)) / std::log(static_cast<float>(blockSize));
  unsigned int pass = static_cast<unsigned int>(t);
  if (std::fabs(t - static_cast<float>(pass)) < 1e-7) pass--;
  if (pass == 0) pass = 1;

  iris::DMem inputBuffer(input, sizeBytes);
  std::vector<std::unique_ptr<float[]>> outputHosts(pass);
  std::vector<std::unique_ptr<float[]>> sumHosts(pass);
  std::vector<std::unique_ptr<iris::DMem>> outputBuffers(pass);
  std::vector<std::unique_ptr<iris::DMem>> blockSumBuffers(pass);

  for (unsigned int i = 0; i < pass; i++) {
    int size = static_cast<int>(length / std::pow(static_cast<float>(blockSize), static_cast<float>(i)));
    outputHosts[i].reset(new float[size]);
    std::memset(outputHosts[i].get(), 0, static_cast<size_t>(size) * sizeof(float));
    outputBuffers[i].reset(new iris::DMem(outputHosts[i].get(), static_cast<size_t>(size) * sizeof(float)));
  }
  for (unsigned int i = 0; i < pass; i++) {
    int size = static_cast<int>(length / std::pow(static_cast<float>(blockSize), static_cast<float>(i + 1)));
    sumHosts[i].reset(new float[size]);
    std::memset(sumHosts[i].get(), 0, static_cast<size_t>(size) * sizeof(float));
    blockSumBuffers[i].reset(new iris::DMem(sumHosts[i].get(), static_cast<size_t>(size) * sizeof(float)));
  }

  int tempLength = static_cast<int>(length / std::pow(static_cast<float>(blockSize), static_cast<float>(pass)));
  std::unique_ptr<float[]> tempHost(new float[tempLength]);
  std::memset(tempHost.get(), 0, static_cast<size_t>(tempLength) * sizeof(float));
  iris::DMem tempBuffer(tempHost.get(), static_cast<size_t>(tempLength) * sizeof(float));

  std::cout << "Executing kernel for " << iterations << " iterations\n";
  std::cout << "-------------------------------------------\n";
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < iterations; n++) {
    bScan(blockSize, static_cast<unsigned int>(length), inputBuffer, *outputBuffers[0],
          *blockSumBuffers[0]);
    for (int i = 1; i < static_cast<int>(pass); i++) {
      int size = static_cast<int>(length / std::pow(static_cast<float>(blockSize), static_cast<float>(i)));
      bScan(blockSize, static_cast<unsigned int>(size), *blockSumBuffers[i - 1], *outputBuffers[i],
            *blockSumBuffers[i]);
    }
    pScan(blockSize, static_cast<unsigned int>(tempLength), *blockSumBuffers[pass - 1], tempBuffer);
    bAddition(blockSize,
              static_cast<unsigned int>(length /
                                        std::pow(static_cast<float>(blockSize), static_cast<float>(pass - 1))),
              tempBuffer, *outputBuffers[pass - 1]);
    for (int i = static_cast<int>(pass) - 1; i > 0; i--) {
      bAddition(blockSize,
                static_cast<unsigned int>(length /
                                          std::pow(static_cast<float>(blockSize), static_cast<float>(i - 1))),
                *outputBuffers[i], *outputBuffers[i - 1]);
    }
  }

  iris::Task flush;
  flush.flush_out(*outputBuffers[0]);
  flush.submit(iris_gpu, nullptr, 1);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of scan kernels: " << time * 1e-3f / iterations
            << " (us)\n";

  std::memcpy(output, outputHosts[0].get(), sizeBytes);
  float* verificationOutput = static_cast<float*>(std::malloc(sizeBytes));
  std::memset(verificationOutput, 0, sizeBytes);
  scanLargeArraysCPUReference(verificationOutput, input, static_cast<unsigned int>(length));
  std::cout << (compare<float>(output, verificationOutput, length, 0.001f) ? "PASS" : "FAIL")
            << std::endl;

  std::free(input);
  std::free(output);
  std::free(verificationOutput);
  platform.finalize();
  return 0;
}
