#include "util.h"
const char inputName128[] = "data/input_14_1_128.bin";
const char biasName128[] = "data/bias_128.bin";
const char weight_winograd_Name128[] = "data/weight_winograd_128_128.bin";
const char bnBias_winograd_Name128[] = "data/bnBias_winograd_128.bin";
const char bnScale_winograd_Name128[] = "data/bnScale_winograd_128.bin";

void kernel_128(double &time, double &ktime) {
  float *input_ = get_parameter(inputName128, 16*16*128);
  float *bias = get_parameter(biasName128, 128);

  float *kernel = get_parameter(weight_winograd_Name128, 36*128*128);
  float *bnBias, *bnScale;

  int nInput = 16*16*128, nOutput = 16*16*128, nWeights = 36*128*128, nBias = 128,
      nTransInput = 16*6*6*128, nInnerProd = 16*6*6*128;

  float result[nOutput];

  bnBias = get_parameter(bnBias_winograd_Name128, 128);
  bnScale = get_parameter(bnScale_winograd_Name128, 128);

  auto start = std::chrono::steady_clock::now();

  std::vector<float> host_input((nInput<<2)/sizeof(float), 0.0f);
  iris::DMem input(host_input.data(), nInput<<2);
  std::vector<float> host_output((nOutput<<2)/sizeof(float), 0.0f);
  iris::DMem output(host_output.data(), nOutput<<2);
  std::vector<float> host_l_weights((nWeights<<2)/sizeof(float), 0.0f);
  iris::DMem l_weights(host_l_weights.data(), nWeights<<2);
  std::vector<float> host_t_input((nTransInput<<2)/sizeof(float), 0.0f);
  iris::DMem t_input(host_t_input.data(), nTransInput<<2);
  std::vector<float> host_ip((nInnerProd<<2)/sizeof(float), 0.0f);
  iris::DMem ip(host_ip.data(), nInnerProd<<2);


  if (input.update(input_) != IRIS_SUCCESS) exit(1);
  if (l_weights.update(kernel) != IRIS_SUCCESS) exit(1);

  std::vector<float> host_l_bnBias((nBias<<2)/sizeof(float), 0.0f);
  iris::DMem l_bnBias(host_l_bnBias.data(), nBias<<2);
  std::vector<float> host_l_bnScale((nBias<<2)/sizeof(float), 0.0f);
  iris::DMem l_bnScale(host_l_bnScale.data(), nBias<<2);
  if (l_bnBias.update(bnBias) != IRIS_SUCCESS) exit(1);
  if (l_bnScale.update(bnScale) != IRIS_SUCCESS) exit(1);

  auto kstart = std::chrono::steady_clock::now();

  launch("kernel_128_winograd_BtdB", {4, 4}, {128, 6}, {&input, &t_input});
  launch("kernel_128_OuterProduct_128", {36, 2}, {128, 8}, {&t_input, &l_weights, &ip});
  launch("kernel_128_winograd_AtIA", {4, 4, 128}, {6, 6}, {&ip, &l_bnBias, &l_bnScale, &output});

  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  if (output.fetch(result) != IRIS_SUCCESS) exit(1);
  dump_result(result, nOutput);


  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  #ifdef DEBUG
  double s = 0;
  for (int i = 0; i < nOutput; i++) {
    s += result[i];
  }
  printf("Check sum: %lf\n", s);
  #endif

  free(kernel);
  free(bnScale);
  free(bnBias);
  free(bias);
  free(input_);
}

