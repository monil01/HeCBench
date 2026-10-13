#include "util.h"
const char inputName256[] = "data/input_14_1_256.bin";
const char weight_winograd_Name256[] = "data/weight_winograd_256_256.bin";
const char bnBias_winograd_Name256[] = "data/bnBias_winograd_256.bin";
const char bnScale_winograd_Name256[] = "data/bnScale_winograd_256.bin";

void kernel_256(double &time, double &ktime) {
  float *input_ = get_parameter(inputName256, 16*16*256);

  float *kernel = get_parameter(weight_winograd_Name256, 36*256*256);
  int nInput = 16*16*256, nOutput = 16*16*256, nWeights = 36*256*256, nBias = 256,
      nTransInput = 16*6*6*256, nInnerProd = 16*6*6*256;
  float *bnBias, *bnScale;

  float result[nOutput];
  bnBias = get_parameter(bnBias_winograd_Name256, 256);
  bnScale = get_parameter(bnScale_winograd_Name256, 256);

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
  std::vector<float> host_l_bnBias((nBias<<2)/sizeof(float), 0.0f);
  iris::DMem l_bnBias(host_l_bnBias.data(), nBias<<2);
  std::vector<float> host_l_bnScale((nBias<<2)/sizeof(float), 0.0f);
  iris::DMem l_bnScale(host_l_bnScale.data(), nBias<<2);


  if (input.update(input_) != IRIS_SUCCESS) exit(1);
  if (l_weights.update(kernel) != IRIS_SUCCESS) exit(1);
  if (l_bnBias.update(bnBias) != IRIS_SUCCESS) exit(1);
  if (l_bnScale.update(bnScale) != IRIS_SUCCESS) exit(1);

  auto kstart = std::chrono::steady_clock::now();

  launch("kernel_256_winograd_BtdB", {4, 4, 2}, {128, 6}, {&input, &t_input});
  launch("kernel_256_OuterProduct_256", {36, 2}, {256, 4}, {&t_input, &l_weights, &ip});
  launch("kernel_256_winograd_AtIA", {4, 4, 256}, {6, 6}, {&ip, &l_bnBias, &l_bnScale, &output});

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
  free(input_);
}

