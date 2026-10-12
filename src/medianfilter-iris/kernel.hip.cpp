#include <hip/hip_runtime.h>

extern "C" __global__ void ckMedian(const uchar4* uc4Source,
                                     unsigned int* uiDest,
                                     const int iLocalPixPitch,
                                     const int iImageWidth,
                                     const int iDevImageHeight)
{
  int iLocalIdX = threadIdx.x;
  int iLocalIdY = threadIdx.y;
  int iGroupIdX = blockIdx.x;
  int iGroupIdY = blockIdx.y;
  int iBlockX = blockDim.x;
  int iBlockY = blockDim.y;
  int iImagePosX = iGroupIdX * iBlockX + iLocalIdX;
  int iDevYPrime = iGroupIdY * iBlockY + iLocalIdY - 1;
  int iImageX = iImageWidth;
  __shared__ uchar4 uc4LocalData[18 * 6];
  int iDevGMEMOffset = iDevYPrime * iImageX + iImagePosX;
  int iLocalPixOffset = iLocalIdY * iLocalPixPitch + iLocalIdX + 1;

  if ((iDevYPrime > -1) && (iDevYPrime < iDevImageHeight) && (iImagePosX < iImageWidth))
    uc4LocalData[iLocalPixOffset] = uc4Source[iDevGMEMOffset];
  else
    uc4LocalData[iLocalPixOffset] = make_uchar4(0, 0, 0, 0);

  if (iLocalIdY < 2) {
    iLocalPixOffset += iBlockY * iLocalPixPitch;
    if (((iDevYPrime + iBlockY) < iDevImageHeight) && (iImagePosX < iImageWidth))
      uc4LocalData[iLocalPixOffset] = uc4Source[iDevGMEMOffset + iBlockY * iImageX];
    else
      uc4LocalData[iLocalPixOffset] = make_uchar4(0, 0, 0, 0);
  }

  if (iLocalIdX == (iBlockX - 1)) {
    iLocalPixOffset = iLocalIdY * iLocalPixPitch;
    if ((iDevYPrime > -1) && (iDevYPrime < iDevImageHeight) && (iGroupIdX > 0))
      uc4LocalData[iLocalPixOffset] = uc4Source[iDevYPrime * iImageX + iGroupIdX * iBlockX - 1];
    else
      uc4LocalData[iLocalPixOffset] = make_uchar4(0, 0, 0, 0);
    if (iLocalIdY < 2) {
      iLocalPixOffset += iBlockY * iLocalPixPitch;
      if (((iDevYPrime + iBlockY) < iDevImageHeight) && (iGroupIdX > 0))
        uc4LocalData[iLocalPixOffset] =
            uc4Source[(iDevYPrime + iBlockY) * iImageX + iGroupIdX * iBlockX - 1];
      else
        uc4LocalData[iLocalPixOffset] = make_uchar4(0, 0, 0, 0);
    }
  } else if (iLocalIdX == 0) {
    iLocalPixOffset = (iLocalIdY + 1) * iLocalPixPitch - 1;
    if ((iDevYPrime > -1) && (iDevYPrime < iDevImageHeight) &&
        ((iGroupIdX + 1) * iBlockX < iImageWidth))
      uc4LocalData[iLocalPixOffset] =
          uc4Source[iDevYPrime * iImageX + (iGroupIdX + 1) * iBlockX];
    else
      uc4LocalData[iLocalPixOffset] = make_uchar4(0, 0, 0, 0);
    if (iLocalIdY < 2) {
      iLocalPixOffset += iBlockY * iLocalPixPitch;
      if (((iDevYPrime + iBlockY) < iDevImageHeight) &&
          ((iGroupIdX + 1) * iBlockX < iImageWidth))
        uc4LocalData[iLocalPixOffset] =
            uc4Source[(iDevYPrime + iBlockY) * iImageX + (iGroupIdX + 1) * iBlockX];
      else
        uc4LocalData[iLocalPixOffset] = make_uchar4(0, 0, 0, 0);
    }
  }

  __syncthreads();

  float fMedianEstimate[3] = {128.0f, 128.0f, 128.0f};
  float fMinBound[3] = {0.0f, 0.0f, 0.0f};
  float fMaxBound[3] = {255.0f, 255.0f, 255.0f};
  for (int iSearch = 0; iSearch < 8; iSearch++) {
    unsigned int uiHighCount[3] = {0, 0, 0};
    iLocalPixOffset = iLocalIdY * iLocalPixPitch + iLocalIdX;
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        uchar4 pix = uc4LocalData[iLocalPixOffset + col];
        uiHighCount[0] += (fMedianEstimate[0] < pix.x);
        uiHighCount[1] += (fMedianEstimate[1] < pix.y);
        uiHighCount[2] += (fMedianEstimate[2] < pix.z);
      }
      iLocalPixOffset += iLocalPixPitch;
    }
    for (int c = 0; c < 3; c++) {
      if (uiHighCount[c] > 4)
        fMinBound[c] = fMedianEstimate[c];
      else
        fMaxBound[c] = fMedianEstimate[c];
      fMedianEstimate[c] = 0.5f * (fMaxBound[c] + fMinBound[c]);
    }
  }

  unsigned int uiPackedPix = 0x000000FF & (unsigned int)(fMedianEstimate[0] + 0.5f);
  uiPackedPix |= 0x0000FF00 & (((unsigned int)(fMedianEstimate[1] + 0.5f)) << 8);
  uiPackedPix |= 0x00FF0000 & (((unsigned int)(fMedianEstimate[2] + 0.5f)) << 16);
  if ((iDevYPrime < iDevImageHeight) && (iImagePosX < iImageWidth))
    uiDest[iDevGMEMOffset + iImageX] = uiPackedPix;
}
