#include "util.h"
const char inputName128one[] = "data/input_one_14_1024.bin";
const char weightName128one[] = "data/weight_one_1024.bin";
const char bnBias_myKernel_Name128one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name128one[] = "data/bnScale_myKernel_one_1024.bin";

void kernel_128_1_in(double &time, double &ktime) {
  float *input = get_parameter(inputName128one, 14*14*512);
  float *weight = get_parameter(weightName128one, 128*512);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name128one, 128);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name128one, 128);


  int nInput = 14*14*512, nOutput = 14*14*128, nWeights = 128*512;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  std::vector<float> host_input_((nInput<<2)/sizeof(float), 0.0f);
  iris::DMem input_(host_input_.data(), nInput<<2);
  std::vector<float> host_output_((nOutput<<2)/sizeof(float), 0.0f);
  iris::DMem output_(host_output_.data(), nOutput<<2);
  std::vector<float> host_weight_((nWeights<<2)/sizeof(float), 0.0f);
  iris::DMem weight_(host_weight_.data(), nWeights<<2);
  std::vector<float> host_bnBias_((128<<2)/sizeof(float), 0.0f);
  iris::DMem bnBias_(host_bnBias_.data(), 128<<2);
  std::vector<float> host_bnScale_((128<<2)/sizeof(float), 0.0f);
  iris::DMem bnScale_(host_bnScale_.data(), 128<<2);

  if (input_.update(input) != IRIS_SUCCESS) exit(1);
  if (weight_.update(weight) != IRIS_SUCCESS) exit(1);
  if (bnBias_.update(bnBias_myKernel) != IRIS_SUCCESS) exit(1);
  if (bnScale_.update(bnScale_myKernel) != IRIS_SUCCESS) exit(1);

  auto kstart = std::chrono::steady_clock::now();

  launch("kernel_512_one_128", {49}, {128, 4}, {&input_, &weight_, &bnBias_, &bnScale_, &output_});

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

void kernel_128_1_out(double &time, double &ktime) {
  float *input = get_parameter(inputName128one, 14*14*128);
  float *weight = get_parameter(weightName128one, 128*512);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name128one, 512);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name128one, 512);


  int nInput = 14*14*128, nOutput = 14*14*512, nWeights = 128*512;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  std::vector<float> host_input_((nInput<<2)/sizeof(float), 0.0f);
  iris::DMem input_(host_input_.data(), nInput<<2);
  std::vector<float> host_output_((nOutput<<2)/sizeof(float), 0.0f);
  iris::DMem output_(host_output_.data(), nOutput<<2);
  std::vector<float> host_weight_((nWeights<<2)/sizeof(float), 0.0f);
  iris::DMem weight_(host_weight_.data(), nWeights<<2);
  std::vector<float> host_bnBias_((512<<2)/sizeof(float), 0.0f);
  iris::DMem bnBias_(host_bnBias_.data(), 512<<2);
  std::vector<float> host_bnScale_((512<<2)/sizeof(float), 0.0f);
  iris::DMem bnScale_(host_bnScale_.data(), 512<<2);

  if (input_.update(input) != IRIS_SUCCESS) exit(1);
  if (weight_.update(weight) != IRIS_SUCCESS) exit(1);
  if (bnBias_.update(bnBias_myKernel) != IRIS_SUCCESS) exit(1);
  if (bnScale_.update(bnScale_myKernel) != IRIS_SUCCESS) exit(1);

  auto kstart = std::chrono::steady_clock::now();

  launch("kernel_128_one_512", {49, 4}, {128, 4}, {&input_, &weight_, &bnBias_, &bnScale_, &output_});

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

