#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef unsigned char uchar;

#include "../base64e-cuda/reference.h"

#define EXPECTED_OVERLAP 30.0f
#define PASS_FACTOR 0.60f
#define RETRIES_ON_FAILURE 1

#define BASE_ARRAY_LENGTH 33554432
#define BASE_LOOP_COUNT 32
#define nStreams 4

static uchar* input = nullptr;
static uchar* output = nullptr;
static uchar* expected_output = nullptr;
static size_t szGlobalWorkSize;
static size_t szLocalWorkSize;

static size_t num_blocks(size_t numElements) {
  return (numElements + 2) / 3;
}

static double RunSequence(iris::DMem& mem_input, iris::DMem& mem_output,
                          int iCycles, size_t numElements,
                          bool multiQueue, bool bShowConfig) {
  const char padCount = static_cast<char>(numElements % 3);
  const size_t numBlock = num_blocks(numElements);
  const size_t streamSize = multiQueue ? (numBlock + nStreams - 1) / nStreams : numBlock;
  szLocalWorkSize = std::min(szLocalWorkSize, streamSize);
  szGlobalWorkSize = (streamSize + szLocalWorkSize - 1) / szLocalWorkSize;
  size_t globalWorkItems = szGlobalWorkSize * szLocalWorkSize;

  for (size_t i = 0; i < numElements; i++) {
    input[i] = static_cast<uchar>(rand() % 256);
  }

  if (multiQueue) {
    const size_t streamBytes = streamSize * 3;
    const size_t streamBytes2 = streamSize * 4;
    printf("Number of blocks = %lu\n", numBlock);
    printf("Global Size = %lu\n\n", szGlobalWorkSize);
    printf("Workgroup Size = %lu\n\n", szLocalWorkSize);
    printf("There are %d streams\n", nStreams);
    printf("Input stream size = %lu  (%lu bytes)\n", streamSize, streamBytes);
    printf("Outputput stream size = %lu  (%lu bytes)\n", streamSize, streamBytes2);
  }

  printf("Run the sequence %d times\n", iCycles);
  double dAvgTime = DBL_MAX;
  for (int c = 0; c < iCycles; c++) {
    auto start = std::chrono::steady_clock::now();
    mem_input.update(input);

    if (multiQueue) {
      std::vector<iris::Task> tasks(nStreams);
      for (int i = 0; i < nStreams; i++) {
        const size_t offset = static_cast<size_t>(i) * streamSize;
        void* params[] = {&mem_input, &mem_output, const_cast<char*>(&padCount),
                          const_cast<size_t*>(&numBlock), const_cast<size_t*>(&offset)};
        int pinfo[] = {iris_r, iris_w, sizeof(padCount), sizeof(numBlock), sizeof(offset)};
        tasks[i].kernel("base64_enc", 1, nullptr, &globalWorkItems, &szLocalWorkSize, 5, params, pinfo);
        tasks[i].flush_out(mem_output);
        tasks[i].submit(iris_gpu, nullptr, 0);
      }
      iris_synchronize();
    } else {
      const size_t offset = 0;
      iris::Task task;
      void* params[] = {&mem_input, &mem_output, const_cast<char*>(&padCount),
                        const_cast<size_t*>(&numBlock), const_cast<size_t*>(&offset)};
      int pinfo[] = {iris_r, iris_w, sizeof(padCount), sizeof(numBlock), sizeof(offset)};
      task.kernel("base64_enc", 1, nullptr, &globalWorkItems, &szLocalWorkSize, 5, params, pinfo);
      task.flush_out(mem_output);
      task.submit(iris_gpu, nullptr, 1);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    dAvgTime = std::min(dAvgTime, time * 1e-9);
  }

  if (bShowConfig) {
    printf("\n%s sequence Configuration:\n", multiQueue ? "Multi-Queue" : "1-Queue");
  }
  return dAvgTime;
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int iWarmupCycles = 4;
  uint uiWorkGroup = 256;
  uint uiSize = BASE_ARRAY_LENGTH;
  uint uiTestCycle = 100;
  bool bPassFlag = false;
  int status = false;
  bool bTestOverlap = true;
  double dAvgGPUTime[2] = {0.0, 0.0};
  double dHostTime[2] = {0.0, 0.0};
  float fMinPassCriteria[2] = {0.0f, 0.0f};

  const uint uiNumElements = uiSize;
  const uint iTestCycles = uiTestCycle;
  const size_t n = uiSize;
  const size_t n2 = uiSize > 1 ? uiSize * 2 + 1 : 5;
  const size_t numBlock = num_blocks(n);
  const size_t streamSize = (numBlock + nStreams - 1) / nStreams;
  const size_t streamBytes = streamSize * 3;
  const size_t bytes = streamBytes * nStreams;
  const size_t bytes2 = n2;

  printf("Input array sizes (without padding) = %zu input elements\n", n);
  printf("Output array sizes = %zu output elements\n", n2);

  szLocalWorkSize = uiWorkGroup;
  fMinPassCriteria[0] = PASS_FACTOR * EXPECTED_OVERLAP;

  input = static_cast<uchar*>(malloc(bytes));
  output = static_cast<uchar*>(malloc(bytes2));
  expected_output = static_cast<uchar*>(malloc(bytes2));
  if (!input || !output || !expected_output) {
    fprintf(stderr, "Host allocation failed\n");
    free(input);
    free(output);
    free(expected_output);
    platform.finalize();
    return 1;
  }

  memset(input, 0, bytes);
  memset(output, 0, bytes2);
  memset(expected_output, 0, bytes2);

  iris::DMem mem_input(input, bytes);
  iris::DMem mem_output(output, bytes2);

  printf("CreateBuffer (Input and Result IRIS DMem, size in byte: %zu %zu)...\n\n", bytes, bytes2);

  size_t encSize;
  printf("Warmup with %d-Queue sequence, %d cycles...\n", nStreams, iWarmupCycles);
  RunSequence(mem_input, mem_output, iWarmupCycles, uiNumElements, true, false);

  for (int iRun = 0; iRun < RETRIES_ON_FAILURE; ++iRun) {
    printf("*******************************************\n");
    printf("Run and time with multiple command-queues\n");
    printf("*******************************************\n");
    dAvgGPUTime[0] = RunSequence(mem_input, mem_output, iTestCycles, uiNumElements, true, false);

    encSize = 0;
    auto start = std::chrono::steady_clock::now();
    reference(input, n, expected_output, n2, &encSize);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    dHostTime[0] = time * 1e-9;

    printf("  Device vs Host Result Comparison\t: ");
    status = memcmp(expected_output, output, encSize);
    printf("%s\n", status ? "FAIL" : "PASS");
    bPassFlag = (status == 0);

    printf("*******************************************\n");
    printf("Run and time with 1 command queue\n");
    printf("*******************************************\n");
    dAvgGPUTime[1] = RunSequence(mem_input, mem_output, iTestCycles, uiNumElements, false, false);

    encSize = 0;
    start = std::chrono::steady_clock::now();
    reference(input, n, expected_output, n2, &encSize);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    dHostTime[1] = time * 1e-9;

    printf("  Device vs Host Result Comparison\t: ");
    status = memcmp(expected_output, output, encSize);
    printf("%s\n", status ? "FAIL" : "PASS");
    bPassFlag &= (status == 0);

    printf("\nResult Summary:\n");
    for (int i = 0; i < 2; i++) {
      printf("  Min GPU Elapsed Time for %d-Queue execution = %.5f s\n",
             i == 0 ? nStreams : 1, dAvgGPUTime[i]);
      printf("  Max GPU Kernel Throughput for %d-Queue execution = %.2f GB/s\n",
             i == 0 ? nStreams : 1, numBlock * 7 / dAvgGPUTime[i] / 1e9);
      printf("  Avg Host Elapsed Time\t\t\t= %.5f s\n\n", dHostTime[i]);
    }

    double dAvgOverlap = 100.0 * (1.0 - dAvgGPUTime[0] / dAvgGPUTime[1]);
    if (bTestOverlap) {
      bool bAvgOverlapOK = (dAvgOverlap >= fMinPassCriteria[1]);
      if (iRun == RETRIES_ON_FAILURE || bAvgOverlapOK) {
        printf("  Measured and (Acceptable) Avg Overlap\t= %.1f %% (%.1f %%)  -> Measured Overlap is %s\n\n",
               dAvgOverlap, fMinPassCriteria[1], bAvgOverlapOK ? "Acceptable" : "NOT Acceptable");
        printf("ComputeOverlap-Avg, Throughput = %.4f OverlapPercent, Time = %.5f s, Size = %u Elements, Workgroup = %lu\n",
               dAvgOverlap, dAvgGPUTime[0], uiNumElements, szLocalWorkSize);
        bPassFlag &= bAvgOverlapOK;
        break;
      }
    }

    printf("  Measured and (Acceptable) Avg Overlap\t= %.1f %% (%.1f %%)  -> Retry %d more time(s)...\n\n",
           dAvgOverlap, fMinPassCriteria[1], RETRIES_ON_FAILURE - iRun);
  }

  printf("Starting Cleanup...\n\n");
  free(input);
  free(output);
  free(expected_output);
  platform.finalize();

  return bPassFlag ? 0 : 1;
}
