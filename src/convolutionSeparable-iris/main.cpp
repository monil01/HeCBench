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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "iris_support.h"
#include <cassert>
#include "conv.h"

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]); 
    return 1;
  }
  const unsigned int imageW = atoi(argv[1]);
  const unsigned int imageH = atoi(argv[2]);
  const int numIterations = atoi(argv[3]);
  if(!imageW || !imageH || numIterations<1) return 1;
  assert(imageW%128==0 && imageH%64==0);
  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,true),"init");
  int device=iris_hip_device();

  float* h_Kernel    = (float*)malloc(KERNEL_LENGTH * sizeof(float));
  float* h_Input     = (float*)malloc(imageW * imageH * sizeof(float));
  float* h_Buffer    = (float*)malloc(imageW * imageH * sizeof(float));
  float* h_OutputCPU = (float*)malloc(imageW * imageH * sizeof(float));
  float* h_OutputGPU = (float*)malloc(imageW * imageH * sizeof(float));

  srand(2009);
  for(unsigned int i = 0; i < KERNEL_LENGTH; i++)
    h_Kernel[i] = (float)(rand() % 16);

  for(unsigned int i = 0; i < imageW * imageH; i++)
    h_Input[i] = (float)(rand() % 16);

  const size_t bytes=sizeof(float)*imageW*imageH;
  iris::DMem d_Kernel(h_Kernel,sizeof(float)*KERNEL_LENGTH);
  iris::DMem d_Input(h_Input,bytes);
  iris::DMem d_Buffer(h_Buffer,bytes);
  iris::DMem d_Output(h_OutputGPU,bytes);
  auto convolutionRows = [&](iris::DMem& dst, iris::DMem& src, iris::DMem& kernel,
                             int width,int height,int pitch) {
    size_t lws[]={16,4}, gws[]={static_cast<size_t>(width/8),static_cast<size_t>(height)};
    iris::Task task;
    void* args[]={&dst,&src,&kernel,&width,&height,&pitch};
    int info[]={iris_w,iris_r,iris_r,sizeof(width),sizeof(height),sizeof(pitch)};
    iris_check(task.kernel("conv_rows",2,nullptr,gws,lws,6,args,info),"row kernel");
    iris_check(task.submit(device,nullptr,true),"HIP rows");
  };
  auto convolutionColumns = [&](iris::DMem& dst, iris::DMem& src, iris::DMem& kernel,
                                int width,int height,int pitch) {
    size_t lws[]={16,8}, gws[]={static_cast<size_t>(width),static_cast<size_t>(height/8)};
    iris::Task task;
    void* args[]={&dst,&src,&kernel,&width,&height,&pitch};
    int info[]={iris_w,iris_r,iris_r,sizeof(width),sizeof(height),sizeof(pitch)};
    iris_check(task.kernel("conv_cols",2,nullptr,gws,lws,6,args,info),"column kernel");
    iris_check(task.submit(device,nullptr,true),"HIP columns");
  };

  //Just a single run or a warmup iteration
  convolutionRows(
      d_Buffer,
      d_Input,
      d_Kernel,
      imageW,
      imageH,
      imageW);

  convolutionColumns(
      d_Output,
      d_Buffer,
      d_Kernel,
      imageW,
      imageH,
      imageW);

  auto start = std::chrono::steady_clock::now();

  for(int iter = 0; iter < numIterations; iter++) {
    convolutionRows(
        d_Buffer,
        d_Input,
        d_Kernel,
        imageW,
        imageH,
        imageW);

    convolutionColumns(
        d_Output,
        d_Buffer,
        d_Kernel,
        imageW,
        imageH,
        imageW);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / numIterations);

  iris::Task flush;
  iris_check(flush.flush_out(d_Output),"flush output");
  iris_check(flush.submit(device,nullptr,true),"HIP flush");

  printf("Comparing against Host/C++ computation...\n"); 
  convolutionRowHost(h_Buffer, h_Input, h_Kernel, imageW, imageH, KERNEL_RADIUS);
  convolutionColumnHost(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH, KERNEL_RADIUS);
  double sum = 0, delta = 0;
  double L2norm;
  for(unsigned int i = 0; i < imageW * imageH; i++){
    delta += (h_OutputCPU[i] - h_OutputGPU[i]) * (h_OutputCPU[i] - h_OutputGPU[i]);
    sum += h_OutputCPU[i] * h_OutputCPU[i];
  }
  L2norm = sqrt(delta / sum);
  printf("Relative L2 norm: %.3e\n\n", L2norm);

  free(h_OutputGPU);
  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Kernel);

  printf("%s\n", L2norm < 1e-6 ? "PASS" : "FAIL");
  iris_check(platform.finalize(),"finalize");
  return L2norm < 1e-6 ? 0 : 1;
}
