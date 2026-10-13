/*  Copyright (c) 2021-2022 Intel Corporation

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "reference.h"


#include "dmem.hpp"
template <typename scalar_t, typename gscalar_t>
float loss_bwd_kernel(
  scalar_t* log_softmax,
  gscalar_t* grad_output,
  gscalar_t* grad_output_neg,
  int64_t* target, scalar_t* weight,
  int64_t* mask, gscalar_t* grad_predict)
{
  dim3 blocks ( threadX, threadBS );
  dim3 grids ( (H + threadX - 1) / threadX, bs );

  auto start = std::chrono::high_resolution_clock::now();

  port::launch(sizeof(scalar_t)==2 ? "ce_half" : (sizeof(scalar_t)==4 ? "ce_float" : "ce_double"), grids, blocks, log_softmax, grad_output, grad_output_neg,
                              target, weight, mask, grad_predict);
  iris_synchronize();

  auto end = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  return time;
}

// compute cross entropy in the backward phase
template <typename scalar_t, typename gscalar_t>
void LossNLL_BWD(int iterations) {

  vector<double> durations(3, 0.0); // timing

  const size_t grad_predict_size = sizeof(gscalar_t) * PredictShape;
  const size_t log_softmax_size = sizeof(scalar_t) * PredictShape;
  const size_t weight_size = sizeof(scalar_t) * OutputShape;
  const size_t target_size = sizeof(int64_t) * TargetShape;
  const size_t mask_size = sizeof(int64_t) * TargetShape;
  const size_t grad_output_size = sizeof(gscalar_t) * OutputShape;

  gscalar_t* grad_predict;
  port::host_allocate((void**)&grad_predict, grad_predict_size);

  gscalar_t* grad_predict_device_host;
  port::host_allocate((void**)&grad_predict_device_host, grad_predict_size);

  scalar_t* weight;
  port::host_allocate((void**)&weight, weight_size);

  int64_t* target;
  port::host_allocate((void**)&target, target_size);

  int64_t* mask;
  port::host_allocate((void**)&mask, mask_size);

  scalar_t* log_softmax;
  port::host_allocate((void**)&log_softmax, log_softmax_size);

  gscalar_t* grad_output;
  port::host_allocate((void**)&grad_output, grad_output_size);

  gscalar_t* grad_output_neg;
  port::host_allocate((void**)&grad_output_neg, grad_output_size);

  for (int i = 0; i < PredictShape; ++i)
    log_softmax[i] = static_cast<scalar_t>(random_float());
  for (int i = 0; i < OutputShape; ++i)
    grad_output[i] = static_cast<gscalar_t>(random_float());
  for (int i = 0; i < OutputShape; ++i)
    grad_output_neg[i] = static_cast<gscalar_t>(random_float());
  for (int i = 0; i < OutputShape; ++i)
    weight[i] = static_cast<scalar_t>(random_float());
  for (int i = 0; i < TargetShape; ++i)
    target[i] = static_cast<int64_t>(random_int(0, W - 1));
  for (int i = 0; i < TargetShape; ++i)
    mask[i] = static_cast<int64_t>(random_int(0, 1));

  // malloc device memory
  auto start = std::chrono::high_resolution_clock::now();

  gscalar_t* grad_predict_device;
  port::allocate((void**)&grad_predict_device, grad_predict_size);

  scalar_t* weight_device;
  port::allocate((void**)&weight_device, weight_size);

  int64_t* target_device;
  port::allocate((void**)&target_device, target_size);

  int64_t* mask_device;
  port::allocate((void**)&mask_device, mask_size);

  scalar_t* log_softmax_device;
  port::allocate((void**)&log_softmax_device, log_softmax_size);

  gscalar_t* grad_output_device;
  port::allocate((void**)&grad_output_device, grad_output_size);

  gscalar_t* grad_output_neg_device;
  port::allocate((void**)&grad_output_neg_device, grad_output_size);

  port::copy(grad_predict_device, grad_predict, grad_predict_size, port::h2d);
  port::copy(target_device, target, target_size, port::h2d);
  port::copy(mask_device, mask, mask_size, port::h2d);
  port::copy(weight_device, weight, weight_size, port::h2d);
  port::copy(grad_output_device, grad_output, grad_output_size, port::h2d);
  port::copy(grad_output_neg_device, grad_output_neg, grad_output_size, port::h2d);
  port::copy(log_softmax_device, log_softmax, log_softmax_size, port::h2d);

  auto end = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  durations[0] += time;

  int warmup = 10;
  for (int k = 0; k < warmup + iterations; ++k) {

    float duration_gpu = loss_bwd_kernel<scalar_t, gscalar_t>(
      log_softmax_device, grad_output_device, grad_output_neg_device,
      target_device, weight_device, mask_device, grad_predict_device);

    if (k >= warmup) {
      durations[1] += duration_gpu;
    }
  }

  start = std::chrono::high_resolution_clock::now();
  port::copy(grad_predict_device_host, grad_predict_device, grad_predict_size, port::d2h);
  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  durations[0] += time;

  start = std::chrono::high_resolution_clock::now();

  loss_bwd_cpu<scalar_t, gscalar_t>(log_softmax, target, weight, mask, grad_output, grad_output_neg, grad_predict);

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  durations[2] = time;

  verify<gscalar_t>(grad_predict, grad_predict_device_host, bs * W * H);

  std::cout << "GPU device memory allocation and data transfer time (ms) : "
            << (durations[0] / iterations) << std::endl;
  std::cout << "Average GPU kernel time (ms) : "
            << (durations[1] / iterations) << std::endl;
  std::cout << "CPU serial time (ms) : "
            << (durations[2]) << std::endl;

  double allBytes = static_cast<double>(sizeof(scalar_t)) * static_cast<double>(PredictShape * 2.0 + OutputShape * 3.0)
                  + static_cast<double>(sizeof(int64_t)) * static_cast<double>(TargetShape * 2.0);

  printf("BandWidth = %lf (GB / s)\n", allBytes / (durations[1] / iterations / 1000) / 1e9);

  port::release(grad_predict_device);
  port::release(weight_device);
  port::release(target_device);
  port::release(mask_device);
  port::release(log_softmax_device);
  port::release(grad_output_device);
  port::release(grad_output_neg_device);

  port::host_release(grad_predict);
  port::host_release(grad_predict_device_host);
  port::host_release(weight);
  port::host_release(target);
  port::host_release(mask);
  port::host_release(log_softmax);
  port::host_release(grad_output);
  port::host_release(grad_output_neg);
}

int main(int argc, char** argv) {
  port::Session session(&argc, &argv);

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  printf("Tensor size (BatchSize * Width * Height) = %d * %d * %d \n", bs, W, H);

  printf("=========== Data type is FP16 ==========\n");

  LossNLL_BWD<__half, __half>(repeat);

  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");

  printf("=========== Data type is FP32 ==========\n");

  LossNLL_BWD<float, float>(repeat);

  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");

  printf("=========== Data type is FP64 ==========\n");

  LossNLL_BWD<double, double>(repeat);

  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");

  return 0;
}
