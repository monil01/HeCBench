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
#include <memory>
#include <iostream>
#include <iris/iris.hpp>
#include "backend.h"
#include "shrUtils.h"
typedef unsigned int uint;

extern void BoxFilterHost(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage,
                          int uiWidth, int uiHeight, int iRadius, float fScale );


const unsigned int RADIUS = 10;                   // initial radius of 2D box filter mask
const float SCALE = 1.0f/(2.0f * RADIUS + 1.0f);  // precalculated GV rescaling value

inline uint DivUp(const uint a, const uint b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

void BoxFilterGPU(iris::DMem* cmBufIn,iris::DMem* cmBufTmp,iris::DMem* cmBufOut,
 int uiWidth,int uiHeight,int iRadius,float fScale,int iCycles) {
 const int szMaxWorkgroupSize=256;
 int iRadiusAligned=((iRadius+15)/16)*16; unsigned int uiNumOutputPix=64;
 if(szMaxWorkgroupSize < iRadiusAligned+uiNumOutputPix+iRadius)
   uiNumOutputPix=szMaxWorkgroupSize-iRadiusAligned-iRadius;
 size_t rl[]={size_t(iRadiusAligned+uiNumOutputPix+iRadius),1};
 size_t rg[]={size_t(DivUp(uiWidth,uiNumOutputPix))*rl[0],size_t(uiHeight)};
 size_t cl[]={64},cg[]={size_t(DivUp(uiWidth,64))*64};
 auto start=std::chrono::steady_clock::now();
 for(int i=0;i<iCycles;i++) {
   iris::Task row;
   void *rp[]={cmBufIn,cmBufTmp,&uiWidth,&uiHeight,&iRadius,&iRadiusAligned,&fScale,&uiNumOutputPix};
   int ri[]={iris_r,iris_rw,sizeof(int),sizeof(int),sizeof(int),sizeof(int),sizeof(float),sizeof(unsigned int)};
   row.kernel("row_kernel",2,nullptr,rg,rl,8,rp,ri); row.flush_out(*cmBufTmp);
   if(row.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
   iris::Task col;
   void *cp[]={cmBufTmp,cmBufOut,&uiWidth,&uiHeight,&iRadius,&fScale};
   int ci[]={iris_r,iris_rw,sizeof(int),sizeof(int),sizeof(int),sizeof(float)};
   col.kernel("col_kernel",1,nullptr,cg,cl,6,cp,ci); col.flush_out(*cmBufOut);
   if(col.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
 }
 auto end=std::chrono::steady_clock::now();
 printf("Average kernel execution time %f (us)\n",std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()*1e-3/iCycles);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage %s <PPM image> <repeat>\n", argv[0]);
    return 1;
  }
  iris::Platform platform; platform.init(&argc,&argv,1); prove_backend();
  const unsigned int uiMaxImageWidth = 1920;   // Image width
  const unsigned int uiMaxImageHeight = 1080;  // Image height
  unsigned int uiImageWidth = 0;     // Image width
  unsigned int uiImageHeight = 0;    // Image height
  unsigned int* uiInput = NULL;      // Host buffer to hold input image data
  unsigned int* uiTmp = NULL;        // Host buffer to hold intermediate image data
  unsigned int* uiDevOutput = NULL;
  unsigned int* uiHostOutput = NULL;

  bool status = shrLoadPPM4ub(argv[1], (unsigned char**)&uiInput,
                              &uiImageWidth, &uiImageHeight);
  printf("Image Width = %u, Height = %u, bpp = %u, Mask Radius = %u\n",
      uiImageWidth, uiImageHeight, unsigned(sizeof(unsigned int) * 8), RADIUS);
  if (uiImageWidth > uiMaxImageWidth || uiImageHeight > uiMaxImageHeight) {
    printf("Error: Image Dimensions exceed the maximum values");
    status = 0;
  }
  if (!status) {
     free(uiInput);
     return 1;
  }

  printf("Using Local Memory for Row Processing\n\n");

  size_t szBuff = uiImageWidth * uiImageHeight;
  size_t szBuffBytes = szBuff * sizeof (unsigned int);

  // Allocate intermediate and output host image buffers
  uiTmp = (unsigned int*)calloc(szBuff, sizeof(unsigned int));
  uiHostOutput = (unsigned int*)calloc(szBuff, sizeof(unsigned int));
  uiDevOutput = (unsigned int*)malloc(szBuffBytes);

  memset(uiDevOutput,0,szBuffBytes);
  iris::DMem input_mem(uiInput,szBuffBytes),temp_mem(uiTmp,szBuffBytes),output_mem(uiDevOutput,szBuffBytes);
  iris::DMem *cmDevBufIn=&input_mem,*cmDevBufTmp=&temp_mem,*cmDevBufOut=&output_mem;
  const int iCycles = atoi(argv[2]);

  printf("Warmup..\n");
  BoxFilterGPU (cmDevBufIn, cmDevBufTmp, cmDevBufOut,
                uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);

  printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);
  BoxFilterGPU (cmDevBufIn, cmDevBufTmp, cmDevBufOut,
                uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);

  // Copy output from device to host






  // Do filtering on the host
  BoxFilterHost(uiInput, uiTmp, uiHostOutput, uiImageWidth, uiImageHeight, RADIUS, SCALE);

  // Verification
  int error = 0;
  const int64_t start = RADIUS * uiImageWidth;
  const int64_t end  = (int64_t)szBuff - (int64_t)start;
  for (int64_t i = start; i < end; i++) {
    if (uiDevOutput[i] != uiHostOutput[i]) {
      printf("%ld %08x %08x\n", i, uiDevOutput[i], uiHostOutput[i]);
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  if(const char *path=getenv("HECBENCH_VALIDATION_DUMP")) {
    FILE *f=fopen(path,"wb"); if(!f) exit(4);
    if(fwrite(uiDevOutput,sizeof(unsigned int),szBuff,f)!=szBuff) exit(4);
    fclose(f);
  }
  free(uiInput);
  free(uiTmp);
  free(uiDevOutput);
  free(uiHostOutput);
  platform.finalize();
  return 0;
}
