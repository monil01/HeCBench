#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "utils.h"
#include "reference.h"

extern "C" __global__ void update_norm(float *__restrict__ x,
       const float * __restrict__ grad,
       float* l2_norm,
       int m, int n, float lambda, float alpha)
{
  typedef hipcub::BlockReduce<float, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage t;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float s = 0;
  if (i < n) {
    s = BlockReduce(t).Sum(x[i]*x[i]);
    float g = grad[i] / (float)m + lambda * x[i];
    x[i] = x[i] - alpha * g;
  }
  if (threadIdx.x == 0) atomicAdd(l2_norm, s);
}

extern "C" __global__ void compute(
    const float * __restrict__ x,
          float * __restrict__ grad,
    const int   * __restrict__ A_row_ptr,
    const int   * __restrict__ A_col_index,
    const float * __restrict__ A_value,
    const int   * __restrict__ A_y_label,
    float * __restrict__ total_obj_val,
    int   * __restrict__ correct,
    int m )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < m) {
    // Simple sparse matrix multiply x' = A * x
    float xp = 0.f;
    for( int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      xp += A_value[j] * x[A_col_index[j]];
    }

    // compute objective
    float v = logf(1.f + expf(-xp * A_y_label[i]));
    atomicAdd(total_obj_val, v);

    // compute errors
    float prediction = 1.f / (1.f + expf(-xp));
    int t = (prediction >= 0.5f) ? 1 : -1;
    if (A_y_label[i] == t) atomicAdd(correct, 1);

    // compute gradient at x
    float accum = expf(-A_y_label[i] * xp);
    accum = accum / (1.f + accum);
    for(int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      float temp = -accum * A_value[j] * A_y_label[i];
      atomicAdd(&grad[A_col_index[j]], temp);
    }
  }
}

extern "C" __global__ void update_plain(float * __restrict__ x,
       const float * __restrict__ grad,
       int m, int n, float lambda, float alpha)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = grad[i] / (float)m + lambda * x[i];
    x[i] = x[i] - alpha * g;
  }
}
