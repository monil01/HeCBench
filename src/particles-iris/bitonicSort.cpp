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

#include <cstdio>
#include <cassert>
#include "particles.h"

#define LOCAL_SIZE_LIMIT 512U



static unsigned int factorRadix2(unsigned int& log2L, unsigned int L){
  if(!L){
    log2L = 0;
    return 0;
  }else{
    for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
    return L;
  }
}

void bitonicSort(
    iris::DMem& d_dstKey,
    iris::DMem& d_dstVal,
    iris::DMem& d_srcKey,
    iris::DMem& d_srcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir)
{
  if(arrayLength < 2) return;

  //Only power-of-two array lengths are supported so far
  unsigned int log2L;
  unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  dir = (dir != 0);

  size_t localWorkSize, globalWorkSize;

  if(arrayLength <= LOCAL_SIZE_LIMIT)
  {
    assert( (batch * arrayLength) % LOCAL_SIZE_LIMIT == 0 );

    //Launch bitonicSortLocal
    localWorkSize  = LOCAL_SIZE_LIMIT / 2;
    globalWorkSize = batch * arrayLength / 2;

    launch("bitonicSortLocal", globalWorkSize, localWorkSize, d_dstKey,  
        d_dstVal,  
        d_srcKey,  
        d_srcVal,  
        arrayLength,
        dir);
  }
  else
  {
    //Launch bitonicSortLocal1
    localWorkSize  = LOCAL_SIZE_LIMIT / 2;
    globalWorkSize = batch * arrayLength / 2;
    launch("bitonicSortLocal1", globalWorkSize, localWorkSize, d_dstKey,
        d_dstVal,
        d_srcKey,
        d_srcVal);

    for(unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
    {
      for(unsigned stride = size / 2; stride > 0; stride >>= 1)
      {
        if(stride >= LOCAL_SIZE_LIMIT)
        {
          //Launch bitonicMergeGlobal
          localWorkSize  = LOCAL_SIZE_LIMIT / 4;
          globalWorkSize = batch * arrayLength / 2;

          launch("bitonicMergeGlobal", globalWorkSize, localWorkSize, d_dstKey,
              d_dstVal,
              d_dstKey,
              d_dstVal,
              arrayLength,
              size,
              stride,
              dir);
        }
        else
        {
          //Launch bitonicMergeLocal
          localWorkSize  = LOCAL_SIZE_LIMIT / 2;
          globalWorkSize = batch * arrayLength / 2;


          assert(stride < LOCAL_SIZE_LIMIT);
          launch("bitonicMergeLocal", globalWorkSize, localWorkSize, d_dstKey,
              d_dstVal,
              d_dstKey,
              d_dstVal,
              arrayLength,
              size,
              stride,
              dir);
          break;
        }
      }
    }
  }
}

