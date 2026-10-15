/* Copyright 1993-2015 NVIDIA Corporation. All rights reserved.
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <hip/hip_runtime.h>
extern "C" __global__ void SimpleKernel(const float* src, float* dst, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i] * 2.0f;
}
