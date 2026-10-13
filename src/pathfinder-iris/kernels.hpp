/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>


// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
#define IN_RANGE(x, min, max)  ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


void fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
}

extern "C" __global__ void pathfinder (
    const int*__restrict__ gpuWall,
    const int*__restrict__ gpuSrc,
          int*__restrict__ gpuResult,
          int*__restrict__ outputBuffer,
    const int iteration,
    const int theHalo,
    const int borderCols,
    const int cols,
    const int t)
{
  int BLOCK_SIZE = blockDim.x;
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  __shared__ int prev[250];
  __shared__ int result[250];

  // Each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  // calculate the small block size.
  int small_block_cols = BLOCK_SIZE - (iteration*theHalo*2);

  // calculate the boundary for the block according to
  // the boundary of its small block
  int blkX = (small_block_cols*bx) - borderCols;
  int blkXmax = blkX+BLOCK_SIZE-1;

  // calculate the global thread coordination
  int xidx = blkX+tx;

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

  int W = tx-1;
  int E = tx+1;

  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if(IN_RANGE(xidx, 0, cols-1))
  {
    prev[tx] = gpuSrc[xidx];
  }

  __syncthreads();

  bool computed;
  for (int i = 0; i < iteration; i++)
  {
    computed = false;

    if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid )
    {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);

      int index = cols*(t+i)+xidx;
      result[tx] = shortest + gpuWall[index];

      // ===================================================================
      // add debugging info to the debug output buffer...
      if (tx==11 && i==0)
      {
        // set bufIndex to what value/range of values you want to know.
        int bufIndex = gpuSrc[xidx];
        // dont touch the line below.
        outputBuffer[bufIndex] = 1;
      }
      // ===================================================================
    }

    __syncthreads();

    if(i==iteration-1)
    {
      // we are on the last iteration, and thus don't need to
      // compute for the next step.
      break;
    }

    if(computed)
    {
      //Assign the computation range
      prev[tx] = result[tx];
    }
    __syncthreads();
  }

  // update the global memory
  // after the last iteration, only threads coordinated within the
  // small block perform the calculation and switch on "computed"
  if (computed)
  {
    gpuResult[xidx] = result[tx];
  }
}
