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

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void
viterbi (const float*__restrict__ maxProbOld, 
         const float*__restrict__ mtState, 
         const float*__restrict__ mtEmit, 
                 int*__restrict__ obs, 
               float*__restrict__ maxProbNew, 
                 int*__restrict__ path, 
         const int nState,
         const int t)
{
  // find the most probable previous state leading to iState
  int iState = blockDim.x * blockIdx.x + threadIdx.x;
  if (iState < nState) {
    float maxProb = 0.0;
    int maxState = -1;
    for (int preState = 0; preState < nState; preState++) 
    {
      float p = maxProbOld[preState] + mtState[iState*nState + preState];
      if (p > maxProb) 
      {
        maxProb = p;
        maxState = preState;
      }
    }
    maxProbNew[iState] = maxProb + mtEmit[obs[t]*nState+iState];
    path[(t-1)*nState+iState] = maxState;
  }
}
