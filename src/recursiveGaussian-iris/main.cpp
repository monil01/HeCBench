
#define CLAMP_TO_EDGE 
#define MAC

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <cassert>
#include <chrono>
#include "dmem.h"

#include "main.h"
#include "shrUtils.h"

double GPUGaussianFilterRGBA(const unsigned int* uiInput,
                             unsigned int* uiOutput,
                             void* d_BufIn,
                             void* d_BufTmp,
                             void* d_BufOut,
                             const unsigned int uiImageWidth,
                             const unsigned int uiImageHeight, 
                             const GaussParms* pGP)
{
#if USE_SIMPLE_FILTER
  float ema = pGP->ema;
#else
  float a0 = pGP->a0;
  float a1 = pGP->a1;
  float a2 = pGP->a2;
  float a3 = pGP->a3;
  float b1 = pGP->b1;
  float b2 = pGP->b2;
  float coefp = pGP->coefp;
  float coefn = pGP->coefn;
#endif

  size_t szBuffBytes = (size_t)uiImageWidth * uiImageHeight * sizeof (unsigned int);
  copyToDevice(d_BufIn, uiInput, szBuffBytes); 

  auto start = std::chrono::steady_clock::now();
  
  // const int iTransposeBlockDim = 16; // initial height and width dimension of 2D transpose workgroup 
  size_t szGaussLocalWork = 256;
  size_t szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageWidth);
  dim3 g_grid (szGaussGlobalWork / szGaussLocalWork);
  dim3 g_block (szGaussLocalWork);
 
#if USE_SIMPLE_FILTER
  launch2("SimpleRecursiveRGBA",g_grid,g_block,{memory(d_BufIn),memory(d_BufTmp),scalar(static_cast<size_t>(uiImageWidth)),scalar(static_cast<size_t>(uiImageHeight)),scalar(ema)});
#else
        launch2("RecursiveRGBA",g_grid,g_block,{memory(d_BufIn),memory(d_BufTmp),scalar(static_cast<size_t>(uiImageWidth)),scalar(static_cast<size_t>(uiImageHeight)),scalar(a0),scalar(a1),scalar(a2),scalar(a3),scalar(b1),scalar(b2),scalar(coefp),scalar(coefn)});
#endif

  size_t szTransposeGlobalWork[2];
  size_t szTransposeLocalWork[2] = {16, 16};
  // Launch transpose kernel in 1st direction
  szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageWidth); 
  szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageHeight); 
  dim3 t1_grid (szTransposeGlobalWork[0] / szTransposeLocalWork[0], 
                szTransposeGlobalWork[1] / szTransposeLocalWork[1]);
  dim3 t1_block (szTransposeLocalWork[0], szTransposeLocalWork[1]);

  launch2("Transpose",t1_grid,t1_block,{memory(d_BufTmp),memory(d_BufOut),scalar(static_cast<size_t>(uiImageWidth)),scalar(static_cast<size_t>(uiImageHeight))});
    
  // Reset Gaussian global work dimensions and variable args, then process in 2nd dimension
  // note width and height parameters flipped due to transpose
  szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageHeight); 

  dim3 g2_grid (szGaussGlobalWork / szGaussLocalWork);

#if USE_SIMPLE_FILTER
  launch2("SimpleRecursiveRGBA",g2_grid,g_block,{memory(d_BufOut),memory(d_BufTmp),scalar(static_cast<size_t>(uiImageHeight)),scalar(static_cast<size_t>(uiImageWidth)),scalar(ema)});
#else
        launch2("RecursiveRGBA",g2_grid,g_block,{memory(d_BufOut),memory(d_BufTmp),scalar(static_cast<size_t>(uiImageHeight)),scalar(static_cast<size_t>(uiImageWidth)),scalar(a0),scalar(a1),scalar(a2),scalar(a3),scalar(b1),scalar(b2),scalar(coefp),scalar(coefn)});
#endif

  // Reset transpose global work dimensions and variable args 
  // note width and height parameters flipped due to 1st transpose
  szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageHeight); 
  szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageWidth); 

  dim3 t2_grid (szTransposeGlobalWork[0] / szTransposeLocalWork[0] , 
                szTransposeGlobalWork[1] / szTransposeLocalWork[1]);
  //range<1> t2_lws (szTransposeLocalWork[1], szTransposeLobalWork[0]);
  // Launch transpose kernel in 2nd direction
  launch2("Transpose",t2_grid,t1_block,{memory(d_BufTmp),memory(d_BufOut),scalar(static_cast<size_t>(uiImageHeight)),scalar(static_cast<size_t>(uiImageWidth))});

  
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  copyFromDevice(uiOutput, d_BufOut, szBuffBytes); 
  return time;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <path to image> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  if(platform.init(&argc,&argv,true)!=IRIS_SUCCESS)return 1;
  int dev=require_amd_hip();if(dev<0)return 1;iris_device_set_default(dev);
  const float fSigma = 10.0f;         // filter sigma (blur factor)
  const int iOrder = 0;               // filter order
  const unsigned int uiMaxImageWidth = 1920;   // Image width
  const unsigned int uiMaxImageHeight = 1080;  // Image height
  unsigned int uiImageWidth;
  unsigned int uiImageHeight;
  unsigned int* uiInput = NULL;       // Host buffer to hold input image data
  unsigned int* uiTemp = NULL;        // Host buffer to hold intermediate image data
  unsigned int* uiOutput = NULL;      // Host buffer to hold output image data

  bool status = shrLoadPPM4ub(argv[1], (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);

  printf("Image Width = %i, Height = %i, bpp = %lu\n\n",
         uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

  if (uiImageWidth > uiMaxImageWidth || uiImageHeight > uiMaxImageHeight) {
    printf("Error: Image Dimensions exceed the maximum values");
    status = 0;
  }
  if (!status) {
     free(uiInput);
     return 1;
  }

  const int iCycles = atoi(argv[2]);

  // Allocate intermediate and output host image buffers
  size_t szBuff = (size_t)uiImageWidth * uiImageHeight;
  size_t szBuffBytes = szBuff * sizeof (unsigned int);
  uiTemp = (unsigned int*)malloc(szBuffBytes);
  uiOutput = (unsigned int*)malloc(szBuffBytes);
  printf("Allocate Host Image Buffers...\n"); 

  // Allocate the source, intermediate and result buffer memory objects on the device GMEM
  void* d_BufIn;
  void* d_BufTmp;
  void* d_BufOut;
  allocDeviceBuffer(&d_BufIn,szBuffBytes);
  allocDeviceBuffer(&d_BufTmp,szBuffBytes);
  allocDeviceBuffer(&d_BufOut,szBuffBytes);

  // init filter coefficients
  PreProcessGaussParms (fSigma, iOrder, &GP);

  // Warmup call to assure OpenCL driver is awake
  GPUGaussianFilterRGBA(uiInput, uiOutput, d_BufIn, d_BufTmp,
                        d_BufOut, uiImageWidth, uiImageHeight, &GP);

  // Start round-trip timer and process iCycles loops on the GPU
  printf("\nRunning GPUGaussianFilterRGBA for %d cycles...\n\n", iCycles);
  double time = 0.0;

  for (int i = 0; i < iCycles; i++)
  {
     time += GPUGaussianFilterRGBA(uiInput, uiOutput, d_BufIn, d_BufTmp,
                                   d_BufOut, uiImageWidth, uiImageHeight, &GP);
  }

  printf("Average execution time of kernels: %f (s)\n", (time * 1e-9f) / iCycles);

  // Compute on host 
  unsigned int* uiGolden = (unsigned int*)malloc(szBuffBytes);
  HostRecursiveGaussianRGBA(uiInput, uiTemp, uiGolden, uiImageWidth, uiImageHeight, &GP);

  printf("Comparing GPU Result to CPU Result...\n"); 
  shrBOOL bMatch = shrCompareuit(uiGolden, uiOutput, szBuff, 1.0f, 0.01f);
  printf("\nGPU Result %s CPU Result within tolerance...\n", (bMatch == shrTRUE) ? "matches" : "DOESN'T match"); 

  if(const char* path=getenv("HECBENCH_DUMP")) {
    FILE* f=fopen(path,"wb");if(!f)return 1;
    if(fwrite(uiOutput,sizeof(unsigned int),szBuff,f)!=szBuff)return 1;
    fclose(f);
  }
  free(uiGolden);
  free(uiInput);
  free(uiTemp);
  free(uiOutput);
  freeDeviceBuffer(d_BufIn);
  freeDeviceBuffer(d_BufTmp);
  freeDeviceBuffer(d_BufOut);
  int errors=platform.error_count();platform.finalize();return bMatch==shrTRUE && !errors?0:1;
}
