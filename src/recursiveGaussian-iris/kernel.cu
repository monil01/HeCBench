#include <cuda_runtime.h>
#include "helper_math.h"
#include "main.h"
// Inline device function to convert 32-bit unsigned integer to floating point rgba color 
//*****************************************************************
__device__
float4 rgbaUintToFloat4(const unsigned int uiPackedRGBA)
{
    float4 rgba;
    rgba.x = uiPackedRGBA & 0xff;
    rgba.y = (uiPackedRGBA >> 8) & 0xff;
    rgba.z = (uiPackedRGBA >> 16) & 0xff;
    rgba.w = (uiPackedRGBA >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************
__device__
unsigned int rgbaFloat4ToUint(const float4 rgba)
{
    unsigned int uiPackedRGBA = 0U;
    uiPackedRGBA |= 0x000000FF & (unsigned int)rgba.x;
    uiPackedRGBA |= 0x0000FF00 & (((unsigned int)rgba.y) << 8);
    uiPackedRGBA |= 0x00FF0000 & (((unsigned int)rgba.z) << 16);
    uiPackedRGBA |= 0xFF000000 & (((unsigned int)rgba.w) << 24);
    return uiPackedRGBA;
}

// Transpose kernel (see transpose SDK sample for details)
//*****************************************************************
extern "C" __global__ 
void Transpose(const unsigned int* uiDataIn, 
                     unsigned int* uiDataOut, 
               const size_t iWidth, const size_t iHeight)
{
    // read the matrix tile into LMEM
    size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ unsigned int uiLocalBuff[16*17];

    if((xIndex < iWidth) && (yIndex < iHeight))
    {
        //uiLocalBuff[get_local_id(1) * (get_local_size(0) + 1) + get_local_id(0)] = uiDataIn[(yIndex * iWidth) + xIndex];
        uiLocalBuff[threadIdx.y * (blockDim.x + 1) + threadIdx.x] = uiDataIn[(yIndex * iWidth) + xIndex];
    }

    // Synchronize the read into LMEM
    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * blockDim.y + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y;
    if((xIndex < iHeight) && (yIndex < iWidth))
    {
        uiDataOut[(yIndex * iHeight) + xIndex] = 
         //uiLocalBuff[get_local_id(0) * (get_local_size(1) + 1) + get_local_id(1)];
         uiLocalBuff[threadIdx.x * (blockDim.y + 1) + threadIdx.y];
    }
}

// 	simple 1st order recursive filter kernel 
//*****************************************************************
//    - processes one image column per thread
//      parameters:	
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data 
//      iWidth  - image width
//      iHeight  - image height
//      a  - blur parameter
//*****************************************************************
extern "C" __global__ void SimpleRecursiveRGBA(
  const unsigned int* uiDataIn,
        unsigned int* uiDataOut,
  const size_t iWidth, const size_t iHeight, const float a)
{
    // compute X pixel location and check in-bounds
  size_t X = blockIdx.x * blockDim.x + threadIdx.x;
  if (X >= iWidth) return;
    
  // advance global pointers to correct column for this work item and x position
  uiDataIn += X;    
  uiDataOut += X;

  // start forward filter pass
  float4 yp = rgbaUintToFloat4(*uiDataIn);  // previous output
  for (int Y = 0; Y < iHeight; Y++) 
  {
    float4 xc = rgbaUintToFloat4(*uiDataIn);
    float4 yc = xc + (yp - xc) * make_float4(a, a, a, a);   
    *uiDataOut = rgbaFloat4ToUint(yc);
    yp = yc;
    uiDataIn += iWidth;     // move to next row
    uiDataOut += iWidth;    // move to next row
  }

  // reset global pointers to point to last element in column for this work item and x position
  uiDataIn -= iWidth;
  uiDataOut -= iWidth;

  // start reverse filter pass: ensures response is symmetrical
  yp = rgbaUintToFloat4(*uiDataIn);
  for (int Y = iHeight - 1; Y > -1; Y--) 
  {
    float4 xc = rgbaUintToFloat4(*uiDataIn);
    float4 yc = xc + (yp - xc) * make_float4(a, a, a, a);
    *uiDataOut = rgbaFloat4ToUint((rgbaUintToFloat4(*uiDataOut) + yc) * 0.5f);
    yp = yc;
    uiDataIn -= iWidth;   // move to previous row
    uiDataOut -= iWidth;  // move to previous row
  }
}

// Recursive Gaussian filter 
//*****************************************************************
//  parameters:	
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data 
//      iWidth  - image width
//      iHeight  - image height
//      a0-a3, b1, b2, coefp, coefn - filter parameters
//
//      If used, CLAMP_TO_EDGE is passed in via OpenCL clBuildProgram call options string at app runtime
//*****************************************************************
extern "C" __global__ void RecursiveRGBA(
  const unsigned int* uiDataIn, 
  unsigned int* uiDataOut, 
  const size_t iWidth, const size_t iHeight, 
  const float a0, const float a1, 
  const float a2, const float a3, 
  const float b1, const float b2, 
  const float coefp, const float coefn)
{
    // compute X pixel location and check in-bounds
    size_t X = blockIdx.x * blockDim.x + threadIdx.x;
    if (X >= iWidth) return;

    // advance global pointers to correct column for this work item and x position
    uiDataIn += X;    
    uiDataOut += X;

    // start forward filter pass
    float4 xp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // previous input
    float4 yp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // previous output
    float4 yb = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // previous output by 2

#ifdef CLAMP_TO_EDGE
    xp = rgbaUintToFloat4(*uiDataIn); 
    yb = xp * make_float4(coefp,coefp,coefp,coefp); 
    yp = yb;
#endif

    for (int Y = 0; Y < iHeight; Y++) 
    {
        float4 xc = rgbaUintToFloat4(*uiDataIn);
        float4 yc = (xc * a0) + (xp * a1) - (yp * b1) - (yb * b2);
		*uiDataOut = rgbaFloat4ToUint(yc);
        xp = xc; 
        yb = yp; 
        yp = yc; 
        uiDataIn += iWidth;     // move to next row
        uiDataOut += iWidth;    // move to next row
    }

    // reset global pointers to point to last element in column for this work item and x position
    uiDataIn -= iWidth;
    uiDataOut -= iWidth;

    // start reverse filter pass: ensures response is symmetrical
    float4 xn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 xa = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 yn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ya = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#ifdef CLAMP_TO_EDGE
    xn = rgbaUintToFloat4(*uiDataIn);
    xa = xn; 
    yn = xn * make_float4(coefn,coefn,coefn,coefn); 
    ya = yn;
#endif

    for (int Y = iHeight - 1; Y > -1; Y--) 
    {
        float4 xc = rgbaUintToFloat4(*uiDataIn);
        float4 yc = (xn * a2) + (xa * a3) - (yn * b1) - (ya * b2);
        xa = xn; 
        xn = xc; 
        ya = yn; 
        yn = yc;
        *uiDataOut = rgbaFloat4ToUint(rgbaUintToFloat4(*uiDataOut) + yc);
        uiDataIn -= iWidth;   // move to previous row
        uiDataOut -= iWidth;  // move to previous row
    }
}
