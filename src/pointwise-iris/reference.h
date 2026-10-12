#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static float LCG_random_ref(unsigned int* seed) {
  const unsigned int m = 2147483648u;
  const unsigned int a = 26757677u;
  const unsigned int c = 1u;
  *seed = (a * (*seed) + c) % m;
  return static_cast<float>(*seed) / static_cast<float>(m);
}

static void init_ref(float* data, int size) {
  for (int index = 0; index < size; index++) {
    unsigned int seed = static_cast<unsigned int>(index) ^ static_cast<unsigned int>(size);
    data[index] = LCG_random_ref(&seed);
  }
}

static void elementwise_ref(int hiddenSize, int miniBatch, const float* tmp_h,
                            const float* tmp_i, const float* bias, float* linearGates,
                            float* h_out, float* i_out, const float* c_in,
                            float* c_out) {
  int numElements = miniBatch * hiddenSize;
  for (int index = 0; index < numElements; index++) {
    int batch = index / hiddenSize;
    int hiddenIdx = index % hiddenSize;
    int gateIndex = hiddenIdx + 4 * batch * hiddenSize;

    float g[4];
    for (int i = 0; i < 4; i++) {
      g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
      g[i] += bias[i * hiddenSize + hiddenIdx] + bias[(i + 4) * hiddenSize + hiddenIdx];
      linearGates[gateIndex + i * hiddenSize] = g[i];
    }

    float in_gate = 1.0f / (1.0f + std::exp(-g[0]));
    float forget_gate = 1.0f / (1.0f + std::exp(-g[1]));
    float in_gate2 = std::tanh(g[2]);
    float out_gate = 1.0f / (1.0f + std::exp(-g[3]));

    float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);
    c_out[index] = val;
    val = out_gate * std::tanh(val);
    h_out[index] = val;
    i_out[index] = val;
  }
}

static void test_ref(int hiddenSize, int miniBatch, int seqLength, int numLayers,
                     float* testOutputi, float* testOutputh, float* testOutputc) {
  int numElements = hiddenSize * miniBatch;
  int hc_size = (seqLength + 1) * numLayers * numElements;
  int i_size = seqLength * (numLayers + 1) * numElements;
  int bias_size = numLayers * hiddenSize * 8;
  int tmp_h_size = 4 * numLayers * numElements;
  int tmp_i_size = 4 * seqLength * numElements;
  int lg_size = 4 * seqLength * numLayers * numElements;

  float* h_data = static_cast<float*>(std::malloc(static_cast<size_t>(hc_size) * sizeof(float)));
  float* i_data = static_cast<float*>(std::malloc(static_cast<size_t>(i_size) * sizeof(float)));
  float* c_data = static_cast<float*>(std::malloc(static_cast<size_t>(hc_size) * sizeof(float)));
  float* bias = static_cast<float*>(std::malloc(static_cast<size_t>(bias_size) * sizeof(float)));
  float* tmp_h = static_cast<float*>(std::malloc(static_cast<size_t>(tmp_h_size) * sizeof(float)));
  float* tmp_i = static_cast<float*>(std::malloc(static_cast<size_t>(tmp_i_size) * sizeof(float)));
  float* linearGates = static_cast<float*>(std::malloc(static_cast<size_t>(lg_size) * sizeof(float)));
  if (!h_data || !i_data || !c_data || !bias || !tmp_h || !tmp_i || !linearGates) {
    std::fprintf(stderr, "malloc failed\n");
    std::exit(1);
  }

  init_ref(tmp_h, tmp_h_size);
  init_ref(tmp_i, tmp_i_size);
  init_ref(c_data, hc_size);
  init_ref(bias, bias_size);
  std::memset(h_data, 0, static_cast<size_t>(hc_size) * sizeof(float));
  std::memset(i_data, 0, static_cast<size_t>(i_size) * sizeof(float));
  std::memset(linearGates, 0, static_cast<size_t>(lg_size) * sizeof(float));

  int lStart = 0, lEnd = 0, rStart = 0, rEnd = 0;
  int recurBatchSize = 2;
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
    for (int layer = lStart; layer < lEnd; layer++) {
      for (int i = rStart; i < rEnd; i++) {
        elementwise_ref(hiddenSize, miniBatch, tmp_h + 4 * layer * numElements,
                        tmp_i + 4 * i * numElements, bias + 8 * layer * hiddenSize,
                        linearGates + 4 * (i * numElements + layer * seqLength * numElements),
                        h_data + (i + 1) * numElements +
                            layer * (seqLength + 1) * numElements,
                        i_data + i * numElements + (layer + 1) * seqLength * numElements,
                        c_data + i * numElements + layer * (seqLength + 1) * numElements,
                        c_data + (i + 1) * numElements +
                            layer * (seqLength + 1) * numElements);
      }
    }
  }

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
