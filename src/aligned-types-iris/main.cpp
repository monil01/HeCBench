#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "types.h"

constexpr int MEM_SIZE = 50000000;
constexpr int NUM_ITERATIONS = 1000;

template <class TData>
int testCPU(TData* h_odata, TData* h_idata, int numElements, int packedElementSize) {
  for (int pos = 0; pos < numElements; pos++) {
    TData src = h_idata[pos];
    TData dst = h_odata[pos];
    for (int i = 0; i < packedElementSize; i++) {
      if (((char*)&src)[i] != ((char*)&dst)[i]) return 0;
    }
  }
  return 1;
}

template <class TData>
int runTest(const char* kernel_name,
            iris::DMem& in_mem,
            iris::DMem& out_mem,
            unsigned char* h_idataCPU,
            unsigned char* h_odataGPU,
            int packedElementSize,
            int memory_size) {
  const int totalMemSizeAligned = iAlignDown(memory_size, sizeof(TData));
  const int numElements = iDivDown(memory_size, sizeof(TData));

  std::memset(h_odataGPU, 0, memory_size);

  size_t gws[] = {static_cast<size_t>(((numElements + 255) / 256) * 256)};
  size_t lws[] = {256};
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    int num_elements_param = numElements;
    iris::Task task;
    void* params[] = {&out_mem, &in_mem, &num_elements_param};
    int pinfo[] = {iris_w, iris_r, sizeof(num_elements_param)};
    task.kernel(kernel_name, 1, nullptr, gws, lws, 3, params, pinfo);
    if (i == NUM_ITERATIONS - 1) task.flush_out(out_mem);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  double gpuTime = elapsed_seconds.count() / NUM_ITERATIONS;

  printf("Avg. time: %f ms / Copy throughput: %f GB/s.\n",
         gpuTime * 1000,
         (double)totalMemSizeAligned / (gpuTime * 1073741824.0));

  int flag = testCPU((TData*)h_odataGPU, (TData*)h_idataCPU, numElements, packedElementSize);
  printf("\tTEST %s\n", flag ? "PASS" : "FAIL");
  return !flag;
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int nTotalFailures = 0;
  printf("[%s] - Starting...\n", argv[0]);
  printf("Allocating memory...\n");

  int MemorySize = (int)(MEM_SIZE)&0xffffff00;
  std::vector<unsigned char> h_idataCPU(MemorySize);
  std::vector<unsigned char> h_odataGPU(MemorySize);

  printf("Generating host input data array...\n");
  for (int i = 0; i < MemorySize; i++) h_idataCPU[i] = (i & 0xFF) + 1;

  iris::DMem in_mem(h_idataCPU.data(), MemorySize);
  iris::DMem out_mem(h_odataGPU.data(), MemorySize);

  printf("Uploading input data to GPU memory...\n");

  printf("Testing misaligned types...\n");
  printf("uchar_misaligned...\n");
  nTotalFailures += runTest<uchar_misaligned>("copy_uchar_misaligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 1, MemorySize);

  printf("uchar4_misaligned...\n");
  nTotalFailures += runTest<uchar4_misaligned>("copy_uchar4_misaligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 4, MemorySize);

  printf("uchar4_aligned...\n");
  nTotalFailures += runTest<uchar4_aligned>("copy_uchar4_aligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 4, MemorySize);

  printf("ushort_misaligned...\n");
  nTotalFailures += runTest<ushort_misaligned>("copy_ushort_misaligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 2, MemorySize);

  printf("uint_aligned...\n");
  nTotalFailures += runTest<uint_aligned>("copy_uint_aligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 4, MemorySize);

  printf("uint2_misaligned...\n");
  nTotalFailures += runTest<uint2_misaligned>("copy_uint2_misaligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 8, MemorySize);

  printf("uint2_aligned...\n");
  nTotalFailures += runTest<uint2_aligned>("copy_uint2_aligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 8, MemorySize);

  printf("uint3_misaligned...\n");
  nTotalFailures += runTest<uint3_misaligned>("copy_uint3_misaligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 12, MemorySize);

  printf("uint3_aligned...\n");
  nTotalFailures += runTest<uint3_aligned>("copy_uint3_aligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 12, MemorySize);

  printf("uint4_misaligned...\n");
  nTotalFailures += runTest<uint4_misaligned>("copy_uint4_misaligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 16, MemorySize);

  printf("uint4_aligned...\n");
  nTotalFailures += runTest<uint4_aligned>("copy_uint4_aligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 16, MemorySize);

  printf("uint8_misaligned...\n");
  nTotalFailures += runTest<uint8_misaligned>("copy_uint8_misaligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 32, MemorySize);

  printf("uint8_aligned...\n");
  nTotalFailures += runTest<uint8_aligned>("copy_uint8_aligned", in_mem, out_mem, h_idataCPU.data(), h_odataGPU.data(), 32, MemorySize);

  printf("\n[alignedTypes] -> Test Results: %d Failures\n", nTotalFailures);
  printf("Shutting down...\n");

  platform.finalize();
  if (nTotalFailures != 0) {
    printf("Test failed!\n");
    return EXIT_FAILURE;
  }

  printf("Test passed\n");
  return EXIT_SUCCESS;
}
