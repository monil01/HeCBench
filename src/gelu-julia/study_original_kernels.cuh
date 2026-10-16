// Verbatim original CUDA device definitions; only host-reference include omitted.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// width is hidden_dim and height is seq_len
__global__ void gelu_bias_loop(__half* src, const __half* bias, int width, int height)
{
  int x     = blockIdx.x;  // seq length
  int y     = threadIdx.x * 2;
  int batch = blockIdx.y;

  int    index = batch * width * height + x * width;
  for (; y < width; y = y + blockDim.x * 2) {
    auto v_bias = ((half2*)bias)[y >> 1];
    auto v_src  = ((half2*)src)[(index + y) >> 1];
    auto v      = __hadd2(v_src, v_bias);
    auto t      = __half22float2(v);
    t.x    = (0.5f * t.x * (1.0f + tanhf(0.79788456f * (t.x + 0.044715f * t.x * t.x * t.x))));
    t.y    = (0.5f * t.y * (1.0f + tanhf(0.79788456f * (t.y + 0.044715f * t.y * t.y * t.y))));
    ((half2*)src)[(index + y) >> 1] = __float22half2_rn(t);
  }
}

__global__ void gelu_bias_loop_base(__half* src, const __half* bias, int width, int height)
{
  int x     = blockIdx.x;  // seq length
  int batch = blockIdx.y;

  int   index = batch * width * height + x * width;
  for (int y = threadIdx.x; y < width; y = y + blockDim.x) {
    auto v_bias = bias[y];
    auto v_src  = src[index + y];
    auto v      = v_src + v_bias;
    auto t      = __half2float(v);
    t      = (0.5f * t * (1.0f + tanhf(0.79788456f * (t + 0.044715f * t * t * t))));
    src[index + y] = __float2half_rn(t);
  }
}
