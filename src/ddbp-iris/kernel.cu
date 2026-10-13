/*
This function reconstruct the 3D volume from projections, based on
the Distance-Driven principle. It works by calculating the overlap
in X and Y axis of the volume and the detector boundaries.
The geometry is for DBT with half cone-beam. All parameters are set
in "ParameterSettings" code.

Reference:
- Branchless Distance Driven Projection and Backprojection,
Samit Basu and Bruno De Man (2006)
- GPU Acceleration of Branchless Distance Driven Projection and
Backprojection, Liu et al (2016)
- GPU-Based Branchless Distance-Driven Projection and Backprojection,
Liu et al (2017)
- A GPU Implementation of Distance-Driven Computed Tomography,
Ryan D. Wagner (2017)
---------------------------------------------------------------------
Copyright (C) <2019>  <Rodrigo de Barros Vimieiro>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Original author: Rodrigo de Barros Vimieiro
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

// thread block size
#define BLOCK_SIZE 256

// integration direction
#define integrateXcoord 1
#define integrateYcoord 0

extern "C" __global__ void pad_projections_kernel(
    double* d_img,
    const int nDetXMap,
    const int nDetYMap,
    const int nElem,
    const int np)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nElem)
    d_img[(np*nDetYMap *nDetXMap) + (gid*nDetYMap)] = 0;
}

extern "C" __global__ void map_boudaries_kernel(
    double* d_pBound,
    const int nElem,
    const double valueLeftBound,
    const double sizeElem,
    const double offset)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nElem)
    d_pBound[gid] = (gid - valueLeftBound) * sizeElem + offset;
}

extern "C" __global__ void rot_detector_kernel(
          double* __restrict__ d_pRdetY,
          double* __restrict__ d_pRdetZ,
    const double* __restrict__ d_pYcoord,
    const double* __restrict__ d_pZcoord,
    const double yOffset,
    const double zOffset,
    const double phi,
    const int nElem)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nElem) {
    // cos and sin are in measured in radians.
    d_pRdetY[gid] = ((d_pYcoord[gid] - yOffset) * cos(phi) - 
                     (d_pZcoord[gid] - zOffset) * sin(phi)) + yOffset;
    d_pRdetZ[gid] = ((d_pYcoord[gid] - yOffset) * sin(phi) +
                     (d_pZcoord[gid] - zOffset) * cos(phi)) + zOffset;
  }
}

extern "C" __global__ void mapDet2Slice_kernel(
           double* __restrict__ const pXmapp,
           double* __restrict__ const pYmapp,
    double tubeX,
    double tubeY,
    double tubeZ,
    const double* __restrict__ const pXcoord,
    const double* __restrict__ const pYcoord,
    const double* __restrict__ const pZcoord,
    const double* __restrict__ const pZSlicecoord,
    const int nDetXMap,
    const int nDetYMap,
    const int nz)
{
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;

  if (px < nDetYMap && py < nDetXMap) {

    const int pos = py * nDetYMap + px;

    pXmapp[pos] = ((pXcoord[py] - tubeX)*(pZSlicecoord[nz] - pZcoord[px]) - 
        (pXcoord[py] * tubeZ) + (pXcoord[py] * pZcoord[px])) / (-tubeZ + pZcoord[px]);

    if (py == 0)
      pYmapp[px] = ((pYcoord[px] - tubeY)*(pZSlicecoord[nz] - pZcoord[px]) -
          (pYcoord[px] * tubeZ) + (pYcoord[px] * pZcoord[px])) / (-tubeZ + pZcoord[px]);
  }
}

extern "C" __global__ void img_integration_kernel(
    double* d_img,
    const int nPixX,
    const int nPixY,
    const bool direction,
    const int offsetX,
    const int offsetY,
    const int nSlices)
{
  /*
     Integration of 2D slices over the whole volume

     (S.1.Integration. - Liu et al(2017))

     Perform an inclusive scan
   */

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int px = tx + offsetX;
  const int py = ty + offsetY;
  const int pz = blockIdx.z * blockDim.z + threadIdx.z;

  if (px >= nPixY || py >= nPixX || pz >= nSlices) return;

  if (direction == integrateXcoord) {

    for (int s = 1; s <= blockDim.y; s *= 2) {

      int spot = ty - s;

      double val = 0;

      if (spot >= 0) {
        val = d_img[(pz*nPixY*nPixX) + (offsetY + spot) * nPixY + px];
      }

      if (spot >= 0) {
        d_img[(pz*nPixY*nPixX) + (py * nPixY) + px] += val;
      }
    }
  }
  else
  {
    for (int s = 1; s <= blockDim.x; s *= 2) {

      int spot = tx - s;

      double val = 0;

      if (spot >= 0) {
        val = d_img[(pz*nPixY*nPixX) + py * nPixY + spot + offsetX];
      }

      if (spot >= 0) {
        d_img[(pz*nPixY*nPixX) + (py * nPixY) + px] += val;
      }
    }
  }
}

extern "C" __global__ void bilinear_interpolation_kernel(
          double* __restrict__ d_sliceI,
    const double* __restrict__ d_pProj,
    const double* __restrict__ d_pObjX,
    const double* __restrict__ d_pObjY,
    const double* __restrict__ d_pDetmX,
    const double* __restrict__ d_pDetmY,
    const int nPixXMap,
    const int nPixYMap,
    const int nDetXMap,
    const int nDetYMap,
    const int nDetX,
    const int nDetY,
    const int np, const int detmOffset) 
{
  d_pDetmX += detmOffset;
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;

  // Make sure we don't try and access memory outside the detector
  // by having any threads mapped there return early
  if (px >= nPixYMap || py >= nPixXMap) return;

  //  S.2. Interpolation - Liu et al (2017)

  // Adjust the mapped coordinates to cross the range of (0-nDetX).*duMap 
  // Divide by pixelSize to get a unitary pixel size
  const double xNormData = nDetX - d_pObjX[py] / d_pDetmX[0];
  const int    xData = floor(xNormData);
  const double alpha = xNormData - xData;

  // Adjust the mapped coordinates to cross the range of (0-nDetY).*dyMap  
  // Divide by pixelSize to get a unitary pixel size
  const double yNormData = (d_pObjY[px] / d_pDetmX[0]) - (d_pDetmY[0] / d_pDetmX[0]);
  const int    yData = floor(yNormData);
  const double beta = yNormData - yData;

  double d00, d01, d10, d11;
  if (((xNormData) >= 0) && ((xNormData) <= nDetX) && ((yNormData) >= 0) && ((yNormData) <= nDetY)) 
    d00 = d_pProj[(np*nDetYMap*nDetXMap) + (xData*nDetYMap + yData)];
  else
    d00 = 0.0;

  if (((xData + 1) > 0) && ((xData + 1) <= nDetX) && ((yNormData) >= 0) && ((yNormData) <= nDetY))
    d10 = d_pProj[(np*nDetYMap*nDetXMap) + ((xData + 1)*nDetYMap + yData)];
  else
    d10 = 0.0;

  if (((xNormData) >= 0) && ((xNormData) <= nDetX) && ((yData + 1) > 0) && ((yData + 1) <= nDetY))
    d01 = d_pProj[(np*nDetYMap*nDetXMap) + (xData*nDetYMap + yData + 1)];
  else
    d01 = 0.0;

  if (((xData + 1) > 0) && ((xData + 1) <= nDetX) && ((yData + 1) > 0) && ((yData + 1) <= nDetY))
    d11 = d_pProj[(np*nDetYMap*nDetXMap) + ((xData + 1)*nDetYMap + yData + 1)];
  else
    d11 = 0.0;

  double result_temp1 = alpha * d10 + (-d00 * alpha + d00);
  double result_temp2 = alpha * d11 + (-d01 * alpha + d01);

  d_sliceI[py * nPixYMap + px] = beta * result_temp2 + (-result_temp1 * beta + result_temp1);
}

extern "C" __global__ void differentiation_kernel(
          double* __restrict__ d_pVolume,
    const double* __restrict__ d_sliceI,
    double tubeX,
    double rtubeY,
    double rtubeZ,
    const double* __restrict__ const d_pObjX,
    const double* __restrict__ const d_pObjY,
    const double* __restrict__ const d_pObjZ,
    const int nPixX,
    const int nPixY,
    const int nPixXMap,
    const int nPixYMap,
    const double du,
    const double dv,
    const double dx,
    const double dy,
    const double dz,
    const int nz) 
{
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;

  /*
     S.3. Differentiation - Eq. 24 - Liu et al (2017)

     Detector integral projection
     ___________
     |_A_|_B_|___|
     |_C_|_D_|___|
     |___|___|___|


     (px,py)
     ________________
     |_A_|__B__|_____|
     |_C_|(0,0)|(0,1)|
     |___|(1,0)|(1,1)|

     Threads are lauched from D up to nPixX (py) and nPixY (px)
     i.e., they are running on the detector image. Thread (0,0) is on D.

     Coordinates on intergal projection:

     A = py * nPixYMap + px
     B = ((py+1) * nPixYMap) + px
     C = py * nPixYMap + px + 1
     D = ((py+1) * nPixYMap) + px + 1
   */

  if (px < nPixY && py < nPixX) {

    const int pos = (nPixX*nPixY*nz) + (py * nPixY) + px;

    int coordA = py * nPixYMap + px;
    int coordB = ((py + 1) * nPixYMap) + px;
    int coordC = coordA + 1;
    int coordD = coordB + 1;

    // x - ray angle in X coord
    double gamma = atan((d_pObjX[py] + (dx / 2.0) - tubeX) / (rtubeZ - d_pObjZ[nz]));

    // x - ray angle in Y coord
    double alpha = atan((d_pObjY[px] + (dy / 2.0) - rtubeY) / (rtubeZ - d_pObjZ[nz]));

    double dA, dB, dC, dD;

    dA = d_sliceI[coordA];
    dB = d_sliceI[coordB];
    dC = d_sliceI[coordC];
    dD = d_sliceI[coordD];

    // Treat border of interpolated integral detector
    if (dC == 0 && dD == 0) {
      dC = dA;
      dD = dB;
    }

    // S.3.Differentiation - Eq. 24 - Liu et al(2017)
    d_pVolume[pos] += ((dD - dC - dB + dA)*(du*dv*dz / (cos(alpha)*cos(gamma)*dx*dy)));
  }
}

extern "C" __global__ void division_kernel(
    double* d_img,
    const int nPixX,
    const int nPixY,
    const int nSlices,
    const int nProj)
{
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  const int pz = blockIdx.z * blockDim.z + threadIdx.z;

  if (px < nPixY && py < nPixX && pz < nSlices) {
    const int pos = (nPixX*nPixY*pz) + (py * nPixY) + px;
    d_img[pos] /= (double) nProj;
  }
}
