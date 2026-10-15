/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
template <typename T>
__device__
void dot_product(const T *__restrict__ a,
                 const T *__restrict__ b,
                       T *__restrict__ d,
                 const size_t n)
{
  size_t iGID = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  T sum = 0;
  for(size_t idx = iGID; idx < n; idx += gridDim.x * blockDim.x) {
    size_t iInOffset = idx * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
      sum += a[iInOffset + i] * b[iInOffset + i];
  }

  using BlockReduce = hipcub::BlockReduce<T, 1024>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T aggregate = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    atomicAdd(d, aggregate);
  }
}

extern "C" __global__ void dot_float(const float*a,const float*b,float*d,size_t n){dot_product<float>(a,b,d,n);}
extern "C" __global__ void dot_double(const double*a,const double*b,double*d,size_t n){dot_product<double>(a,b,d,n);}
