// Helper function to convert float[4] rgba color to 32-bit unsigned integer
__device__
float4 rgbaUintToFloat4(const unsigned int c)
{
  float4 rgba;
  rgba.x = c & 0xff;
  rgba.y = (c >> 8) & 0xff;
  rgba.z = (c >> 16) & 0xff;
  rgba.w = (c >> 24) & 0xff;
  return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
__device__
unsigned int rgbaFloat4ToUint(const float4 rgba, const float fScale)
{
  unsigned int uiPackedPix = 0U;
  uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
  uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
  uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
  uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
  return uiPackedPix;
}

extern "C" __global__ void row_kernel (
    const uchar4* __restrict__ ucSource,
            uint* __restrict__ uiDest,
    const int uiWidth,
    const int uiHeight,
    const int iRadius,
    const int iRadiusAligned,
    const float fScale,
    const unsigned int uiNumOutputPix)
{
  __shared__ uchar4 uc4LocalData[256];

  int lid = threadIdx.x;
  int gidx = blockIdx.x;
  int gidy = blockIdx.y;

  int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
  int globalPosY = gidy;
  int iGlobalOffset = globalPosY * uiWidth + globalPosX;

  // Read global data into LMEM
  if (globalPosX >= 0 && globalPosX < uiWidth)
  {
    uc4LocalData[lid] = ucSource[iGlobalOffset];
  }
  else
    uc4LocalData[lid] = {0, 0, 0, 0};

  __syncthreads();

  if((globalPosX >= 0) && (globalPosX < uiWidth) && (lid >= iRadiusAligned) &&
      (lid < (iRadiusAligned + (int)uiNumOutputPix)))
  {
    // Init summation registers to zero
    float4 f4Sum = {0.0f, 0.0f, 0.0f, 0.0f};

    // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
    int iOffsetX = lid - iRadius;
    int iLimit = iOffsetX + (2 * iRadius) + 1;
    for(; iOffsetX < iLimit; iOffsetX++)
    {
      f4Sum.x += uc4LocalData[iOffsetX].x;
      f4Sum.y += uc4LocalData[iOffsetX].y;
      f4Sum.z += uc4LocalData[iOffsetX].z;
      f4Sum.w += uc4LocalData[iOffsetX].w;
    }

    // Use inline function to scale and convert registers to packed RGBA values in a uchar4,
    // and write back out to GMEM
    uiDest[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
  }
}

extern "C" __global__ void col_kernel (
    const uint* __restrict__ uiSource,
          uint* __restrict__ uiDest,
    const int uiWidth,
    const int uiHeight,
    const int iRadius,
    const float fScale)
{
  int globalPosX = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalPosX >= uiWidth) return;

  const uint* uiInputImage = &uiSource[globalPosX];
  uint* uiOutputImage = &uiDest[globalPosX];

  float4 f4Sum;

  float4 top_color = rgbaUintToFloat4(uiInputImage[0]);
  float4 bot_color = rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);

  f4Sum = top_color *
          make_float4((float)iRadius, (float)iRadius, (float)iRadius, (float)iRadius);
  for (int y = 0; y < iRadius + 1; y++)
  {
    if (y < uiHeight)
      f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
  }
  for(int y = 1; y < iRadius + 1; y++)
  {
    if (y + iRadius < uiHeight) {
      f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
      f4Sum -= top_color;
      uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }
  }

  for(int y = iRadius + 1; y < uiHeight - iRadius; y++)
  {
    if (y + iRadius < uiHeight && y - iRadius >= 0) {
      f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
      f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
      uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }
  }

  for (int y = uiHeight - iRadius; y < uiHeight; y++)
  {
    if (y < uiHeight && y - iRadius >= 0) {
      f4Sum += bot_color;
      f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
      uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }
  }
}
