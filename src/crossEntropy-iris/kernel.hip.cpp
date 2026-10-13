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

__inline__ __device__
float exponent (float x) { return __expf(x); }

__inline__ __device__
double exponent (double x) { return exp(x); }

template <typename scalar_t, typename gscalar_t>
__device__ void loss_bwd (
    const scalar_t* __restrict__ log_softmax,
    const gscalar_t* __restrict__ grad_output,
    const gscalar_t* __restrict__ grad_output_neg,
    const int64_t* __restrict__ target,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ mask,
          gscalar_t* __restrict__ grad_predict)
{
  int local_id_x = threadIdx.x;
  int group_id_bs = blockIdx.y;
  int group_id_x = blockIdx.x;

  int linear_x_id = group_id_x * threadX + local_id_x;

  if (linear_x_id >= H) return;

  int offset2d = group_id_bs * H + linear_x_id;
  int idx = target[offset2d];
  int sum_offset = group_id_bs * W * H + idx * H + linear_x_id;

  gscalar_t tmp_grad;
  if (mask[offset2d])
    tmp_grad = -(grad_output[offset2d] + grad_output_neg[offset2d]);
  else
    tmp_grad = -grad_output[offset2d];

  tmp_grad = tmp_grad * weight[offset2d];

  float sum_value = tmp_grad * log_softmax[sum_offset];

  #pragma unroll
  for (int i = 0; i < W; ++i) {
    int in_offset = group_id_bs * W * H + i * H + linear_x_id;
    float tmp_sfm = exponent(log_softmax[in_offset]) * sum_value;
    float res = 0.f;
    if (i == idx) {
      res = (float)tmp_grad - tmp_sfm;
    }
    else {
      res = -tmp_sfm;
    }
    grad_predict[in_offset] = res;
  }
}

template <>
__device__ void loss_bwd<__half, __half> (
    const __half* __restrict__ log_softmax,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ grad_output_neg,
    const int64_t* __restrict__ target,
    const __half* __restrict__ weight,
    const int64_t* __restrict__ mask,
          __half* __restrict__ grad_predict)
{
  int local_id_x = threadIdx.x;
  int group_id_bs = blockIdx.y;
  int group_id_x = blockIdx.x;

  int linear_x_id = group_id_x * threadX + local_id_x;

  if (linear_x_id >= H) return;

  int offset2d = group_id_bs * H + linear_x_id;
  int idx = target[offset2d];
  int sum_offset = group_id_bs * W * H + idx * H + linear_x_id;

  __half tmp_grad;
  if (mask[offset2d])
    tmp_grad = __hneg(__hadd(grad_output[offset2d] , grad_output_neg[offset2d]));
  else
    tmp_grad = __hneg(grad_output[offset2d]);

  tmp_grad = __hmul(tmp_grad , weight[offset2d]);

  float sum_value = h2f(__hmul(tmp_grad , log_softmax[sum_offset]));

  #pragma unroll
  for (int i = 0; i < W; ++i) {
    int in_offset = group_id_bs * W * H + i * H + linear_x_id;
    float tmp_sfm = h2f(hexp(log_softmax[in_offset])) * sum_value;
    float res = 0.f;
    if (i == idx) {
      res = h2f(tmp_grad) - tmp_sfm;
    }
    else {
      res = -tmp_sfm;
    }
    grad_predict[in_offset] = res;
  }
}


extern "C" __global__ void ce_half(const __half* logp,const __half* grad,const __half* neg,const int64_t* target,const __half* weight,const int64_t* mask,__half* out) { loss_bwd<__half,__half>(logp,grad,neg,target,weight,mask,out); }

extern "C" __global__ void ce_float(const float* logp,const float* grad,const float* neg,const int64_t* target,const float* weight,const int64_t* mask,float* out) { loss_bwd<float,float>(logp,grad,neg,target,weight,mask,out); }

extern "C" __global__ void ce_double(const double* logp,const double* grad,const double* neg,const int64_t* target,const double* weight,const int64_t* mask,double* out) { loss_bwd<double,double>(logp,grad,neg,target,weight,mask,out); }
