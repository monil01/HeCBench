#include "util.h"
const char inputName256one[] = "data/input_one_14_1024.bin";
const char weightName256one[] = "data/weight_one_1024.bin";
const char bnBias_myKernel_Name256one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name256one[] = "data/bnScale_myKernel_one_1024.bin";

void kernel_256_1_in(double &time, double &ktime) {
  float *input = get_parameter(inputName256one, 14*14*1024);
  float *weight = get_parameter(weightName256one, 256*1024);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name256one, 256);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name256one, 256);


  int nInput = 14*14*1024, nOutput = 14*14*256, nWeights = 256*1024;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  std::vector<float> host_input_((nInput<<2)/sizeof(float), 0.0f);
  iris::DMem input_(host_input_.data(), nInput<<2);
  std::vector<float> host_output_((nOutput<<2)/sizeof(float), 0.0f);
  iris::DMem output_(host_output_.data(), nOutput<<2);
  std::vector<float> host_weight_((nWeights<<2)/sizeof(float), 0.0f);
  iris::DMem weight_(host_weight_.data(), nWeights<<2);
  std::vector<float> host_bnBias_((256<<2)/sizeof(float), 0.0f);
  iris::DMem bnBias_(host_bnBias_.data(), 256<<2);
  std::vector<float> host_bnScale_((256<<2)/sizeof(float), 0.0f);
  iris::DMem bnScale_(host_bnScale_.data(), 256<<2);

  if (input_.update(input) != IRIS_SUCCESS) exit(1);
  if (weight_.update(weight) != IRIS_SUCCESS) exit(1);
  if (bnBias_.update(bnBias_myKernel) != IRIS_SUCCESS) exit(1);
  if (bnScale_.update(bnScale_myKernel) != IRIS_SUCCESS) exit(1);

  auto kstart = std::chrono::steady_clock::now();

  launch("kernel_1024_one_256", {49}, {256, 4}, {&input_, &weight_, &bnBias_, &bnScale_, &output_});

  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  if (output_.fetch(result) != IRIS_SUCCESS) exit(1);
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

  free(input);
  free(weight);
  free(bnBias_myKernel);
  free(bnScale_myKernel);
}

void kernel_256_1_out(double &time, double &ktime) {
  float *input = get_parameter(inputName256one, 14*14*256);
  float *weight = get_parameter(weightName256one, 256*1024);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name256one, 1024);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name256one, 1024);


  int nInput = 14*14*256, nOutput = 14*14*1024, nWeights = 256*1024;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  std::vector<float> host_input_((nInput<<2)/sizeof(float), 0.0f);
  iris::DMem input_(host_input_.data(), nInput<<2);
  std::vector<float> host_output_((nOutput<<2)/sizeof(float), 0.0f);
  iris::DMem output_(host_output_.data(), nOutput<<2);
  std::vector<float> host_weight_((nWeights<<2)/sizeof(float), 0.0f);
  iris::DMem weight_(host_weight_.data(), nWeights<<2);
  std::vector<float> host_bnBias_((1024<<2)/sizeof(float), 0.0f);
  iris::DMem bnBias_(host_bnBias_.data(), 1024<<2);
  std::vector<float> host_bnScale_((1024<<2)/sizeof(float), 0.0f);
  iris::DMem bnScale_(host_bnScale_.data(), 1024<<2);

  if (input_.update(input) != IRIS_SUCCESS) exit(1);
  if (weight_.update(weight) != IRIS_SUCCESS) exit(1);
  if (bnBias_.update(bnBias_myKernel) != IRIS_SUCCESS) exit(1);
  if (bnScale_.update(bnScale_myKernel) != IRIS_SUCCESS) exit(1);

  auto kstart = std::chrono::steady_clock::now();

  launch("kernel_256_one_1024", {49, 4}, {256, 4}, {&input_, &weight_, &bnBias_, &bnScale_, &output_});

  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  if (output_.fetch(result) != IRIS_SUCCESS) exit(1);
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

  free(bnBias_myKernel);
  free(bnScale_myKernel);
  free(input);
  free(weight);
}

