#include <hip/hip_runtime.h>
// Copyright (c) 2019 Intel Corporation; SPDX-License-Identifier: MIT
#include <string>
#include "iso2dfd.h"
extern "C" __global__ void iso_2dfd_kernel(
        float*__restrict__ next,
  const float*__restrict__ prev,
  const float*__restrict__ vel, 
  const float dtDIVdxy, const size_t nRows, const size_t nCols)
{
  // Compute global id
  // We can use the get.global.id() function of the item variable
  //   to compute global id. The 2D array is laid out in memory in row major
  //   order.
  size_t gidCol = blockDim.x * blockIdx.x + threadIdx.x;
  size_t gidRow = blockDim.y * blockIdx.y + threadIdx.y;

  if (gidRow < nRows && gidCol < nCols) {

    size_t gid = (gidRow)*nCols + gidCol;

    // Computation to solve wave equation in 2D
    // First check if gid is inside the effective grid (not in halo)
    if ((gidCol >= HALF_LENGTH && gidCol < nCols - HALF_LENGTH) &&
        (gidRow >= HALF_LENGTH && gidRow < nRows - HALF_LENGTH)) {
      // Stencil code to update grid point at position given by global id (gid)
      // New time step for grid point is computed based on the values of the
      //    the immediate neighbors in both the horizontal and vertical
      //    directions, as well as the value of grid point at a previous time step
      float value = 0.f;
      value += prev[gid + 1] - 2.f * prev[gid] + prev[gid - 1];
      value += prev[gid + nCols] - 2.f * prev[gid] + prev[gid - nCols];
      value *= dtDIVdxy * vel[gid];
      next[gid] = 2.f * prev[gid] - next[gid] + value;
    }
  }
}
