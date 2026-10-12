#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "reference.h"

static void* xmalloc(size_t bytes) {
  void* ptr = std::malloc(bytes);
  if (!ptr) {
    std::fprintf(stderr, "malloc failed\n");
    std::exit(1);
  }
  return ptr;
}

static void test(int hiddenSize, int miniBatch, int seqLength, int numLayers,
                 float* testOutputi, float* testOutputh, float* testOutputc, double& time) {
  const int numElements = hiddenSize * miniBatch;
  const int hc_size = (seqLength + 1) * numLayers * numElements;
  const int i_size = seqLength * (numLayers + 1) * numElements;
  const int bias_size = numLayers * hiddenSize * 8;
  const int tmp_h_size = 4 * numLayers * numElements;
  const int tmp_i_size = 4 * seqLength * numElements;
  const int lg_size = 4 * seqLength * numLayers * numElements;

  float* h_data = static_cast<float*>(xmalloc(static_cast<size_t>(hc_size) * sizeof(float)));
  float* i_data = static_cast<float*>(xmalloc(static_cast<size_t>(i_size) * sizeof(float)));
  float* c_data = static_cast<float*>(xmalloc(static_cast<size_t>(hc_size) * sizeof(float)));
  float* bias = static_cast<float*>(xmalloc(static_cast<size_t>(bias_size) * sizeof(float)));
  float* tmp_h = static_cast<float*>(xmalloc(static_cast<size_t>(tmp_h_size) * sizeof(float)));
  float* tmp_i = static_cast<float*>(xmalloc(static_cast<size_t>(tmp_i_size) * sizeof(float)));
  float* linearGates = static_cast<float*>(xmalloc(static_cast<size_t>(lg_size) * sizeof(float)));

  init_ref(tmp_h, tmp_h_size);
  init_ref(tmp_i, tmp_i_size);
  init_ref(c_data, hc_size);
  init_ref(bias, bias_size);
  std::memset(h_data, 0, static_cast<size_t>(hc_size) * sizeof(float));
  std::memset(i_data, 0, static_cast<size_t>(i_size) * sizeof(float));
  std::memset(linearGates, 0, static_cast<size_t>(lg_size) * sizeof(float));

  iris::DMem mem_h(h_data, static_cast<size_t>(hc_size) * sizeof(float));
  iris::DMem mem_i(i_data, static_cast<size_t>(i_size) * sizeof(float));
  iris::DMem mem_c(c_data, static_cast<size_t>(hc_size) * sizeof(float));
  iris::DMem mem_bias(bias, static_cast<size_t>(bias_size) * sizeof(float));
  iris::DMem mem_tmp_h(tmp_h, static_cast<size_t>(tmp_h_size) * sizeof(float));
  iris::DMem mem_tmp_i(tmp_i, static_cast<size_t>(tmp_i_size) * sizeof(float));
  iris::DMem mem_linear(linearGates, static_cast<size_t>(lg_size) * sizeof(float));

  int lStart = 0, lEnd = 0, rStart = 0, rEnd = 0;
  int recurBatchSize = 2;
  size_t gws[1] = {static_cast<size_t>(((numElements + 255) / 256) * 256)};
  size_t lws[1] = {256};
  double ktime = 0.0;

  while (true) {
    if (lEnd == 0) {
      lStart = 0;
      lEnd = 1;
      rStart = 0;
    } else {
      lStart++;
      lEnd++;
      rStart -= recurBatchSize;
      if (lEnd > numLayers || rStart < 0) {
        rStart += (lStart + 1) * recurBatchSize;
        lStart = 0;
        lEnd = 1;
      }
      while (rStart >= seqLength && lEnd <= numLayers) {
        lStart++;
        lEnd++;
        rStart -= recurBatchSize;
      }
      if (lEnd > numLayers || rStart < 0) break;
    }

    rEnd = rStart + recurBatchSize;
    if (rEnd > seqLength) rEnd = seqLength;
    auto start = std::chrono::steady_clock::now();
    for (int layer = lStart; layer < lEnd; layer++) {
      for (int step = rStart; step < rEnd; step++) {
        int tmp_h_offset = 4 * layer * numElements;
        int tmp_i_offset = 4 * step * numElements;
        int bias_offset = 8 * layer * hiddenSize;
        int linear_offset = 4 * (step * numElements + layer * seqLength * numElements);
        int h_offset = (step + 1) * numElements + layer * (seqLength + 1) * numElements;
        int i_offset = step * numElements + (layer + 1) * seqLength * numElements;
        int c_in_offset = step * numElements + layer * (seqLength + 1) * numElements;
        int c_out_offset = (step + 1) * numElements + layer * (seqLength + 1) * numElements;

        void* params[] = {&hiddenSize, &miniBatch, &mem_tmp_h, &tmp_h_offset,
                          &mem_tmp_i, &tmp_i_offset, &mem_bias, &bias_offset,
                          &mem_linear, &linear_offset, &mem_h, &h_offset,
                          &mem_i, &i_offset, &mem_c, &c_in_offset, &c_out_offset};
        int pinfo[] = {sizeof(hiddenSize), sizeof(miniBatch), iris_r, sizeof(tmp_h_offset),
                       iris_r, sizeof(tmp_i_offset), iris_r, sizeof(bias_offset),
                       iris_rw, sizeof(linear_offset), iris_rw, sizeof(h_offset),
                       iris_rw, sizeof(i_offset), iris_rw, sizeof(c_in_offset),
                       sizeof(c_out_offset)};
        iris::Task task;
        task.kernel("elementwise", 1, nullptr, gws, lws, 17, params, pinfo);
        task.flush_out(mem_h);
        task.flush_out(mem_i);
        task.flush_out(mem_c);
        task.submit(iris_gpu, nullptr, 1);
      }
    }
    auto end = std::chrono::steady_clock::now();
    ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  time += ktime;
  std::memcpy(testOutputi, i_data + numLayers * seqLength * numElements,
              static_cast<size_t>(seqLength) * numElements * sizeof(float));
  for (int layer = 0; layer < numLayers; layer++) {
    std::memcpy(testOutputh + layer * numElements,
                h_data + seqLength * numElements + layer * (seqLength + 1) * numElements,
                static_cast<size_t>(numElements) * sizeof(float));
    std::memcpy(testOutputc + layer * numElements,
                c_data + seqLength * numElements + layer * (seqLength + 1) * numElements,
                static_cast<size_t>(numElements) * sizeof(float));
  }

  std::free(h_data);
  std::free(i_data);
  std::free(c_data);
  std::free(bias);
  std::free(tmp_h);
  std::free(tmp_i);
  std::free(linearGates);
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int seqLength, numLayers, hiddenSize, miniBatch, numRuns;
  if (argc == 6) {
    seqLength = std::atoi(argv[1]);
    numLayers = std::atoi(argv[2]);
    hiddenSize = std::atoi(argv[3]);
    miniBatch = std::atoi(argv[4]);
    numRuns = std::atoi(argv[5]);
  } else if (argc == 1) {
    std::printf("Running with default settings\n");
    seqLength = 100;
    numLayers = 4;
    hiddenSize = 512;
    miniBatch = 64;
    numRuns = 1;
  } else {
    std::printf("Usage: %s <seqLength> <numLayers> <hiddenSize> <miniBatch> <repeat>\n", argv[0]);
    platform.finalize();
    return 1;
  }

  std::printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n", seqLength,
              numLayers, hiddenSize, miniBatch);

  int numElements = hiddenSize * miniBatch;
  float* testOutputi =
      static_cast<float*>(xmalloc(static_cast<size_t>(numElements) * seqLength * sizeof(float)));
  float* testOutputh =
      static_cast<float*>(xmalloc(static_cast<size_t>(numElements) * numLayers * sizeof(float)));
  float* testOutputc =
      static_cast<float*>(xmalloc(static_cast<size_t>(numElements) * numLayers * sizeof(float)));
  float* testOutputi_ref =
      static_cast<float*>(xmalloc(static_cast<size_t>(numElements) * seqLength * sizeof(float)));
  float* testOutputh_ref =
      static_cast<float*>(xmalloc(static_cast<size_t>(numElements) * numLayers * sizeof(float)));
  float* testOutputc_ref =
      static_cast<float*>(xmalloc(static_cast<size_t>(numElements) * numLayers * sizeof(float)));

  double time = 0.0;
  for (int run = 0; run < numRuns; run++) {
    test(hiddenSize, miniBatch, seqLength, numLayers, testOutputi, testOutputh, testOutputc, time);
    test_ref(hiddenSize, miniBatch, seqLength, numLayers, testOutputi_ref, testOutputh_ref,
             testOutputc_ref);
  }

  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / numRuns);
  int error = 0;
  for (int m = 0; m < miniBatch; m++) {
    for (int j = 0; j < seqLength; j++) {
      for (int i = 0; i < hiddenSize; i++) {
        if (std::fabs(testOutputi[j * numElements + m * hiddenSize + i] -
                      testOutputi_ref[j * numElements + m * hiddenSize + i]) > 1e-4f) {
          error++;
        }
      }
    }
    for (int j = 0; j < numLayers; j++) {
      for (int i = 0; i < hiddenSize; i++) {
        if (std::fabs(testOutputh[j * numElements + m * hiddenSize + i] -
                      testOutputh_ref[j * numElements + m * hiddenSize + i]) > 1e-4f) {
          error++;
        }
        if (std::fabs(testOutputc[j * numElements + m * hiddenSize + i] -
                      testOutputc_ref[j * numElements + m * hiddenSize + i]) > 1e-4f) {
          error++;
        }
      }
    }
  }

  std::printf("%s\n", (error == 0) ? "PASS" : "FAIL");
  std::free(testOutputi);
  std::free(testOutputh);
  std::free(testOutputc);
  std::free(testOutputi_ref);
  std::free(testOutputh_ref);
  std::free(testOutputc_ref);
  platform.finalize();
  return error == 0 ? 0 : 1;
}
