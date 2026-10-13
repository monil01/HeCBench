/*
ECL-APSP code: ECL-APSP is an all-pairs-shortest-paths CUDA implementation of
the Floyd-Warshall algorithm that is quite fast. It operates on graphs stored
in binary CSR format.

Copyright (c) 2021, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Yiqian Liu and Martin Burtscher

URL: The latest version of this code is available at
https://cs.txstate.edu/~burtscher/research/ECL-APSP/.
*/


#include <chrono>
#include <cstdio>
#include <limits>
#include <hip/hip_runtime.h>
#include "graph.h"

using mtype = int;

static const int ws = 32; // warp size
static const int tile = 64; // tile size
static const int ThreadsPerBlock = ws * ws;

// initialize adj matrix
extern "C" __global__ void init1(
  const int nodes,
  mtype* const AdjMat,
  const int upper)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int i = idx / upper;
  if (i < upper) {
    const int j = idx % upper;
    AdjMat[idx] = ((i == j) && (i < nodes)) ? 0 : (INT_MAX / 2);
  }
}

// add edges to adj matrix
extern "C" __global__ void init2(
  const int nodes,
  const int* nindex,
  const int* nlist,
  const int* eweight,
  mtype* const AdjMat,
  const int upper)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < nodes) {
    for (int j = nindex[i]; j < nindex[i + 1]; j++) {
      const int nei = nlist[j];
      AdjMat[i * upper + nei] = eweight[j];
    }
  }
}

// calculate the first tile
extern "C" __global__ __launch_bounds__(ThreadsPerBlock, 1)
void FW0_64(
  mtype* const __restrict__ AdjMat,
  const int upper,
  mtype* const __restrict__ krows,
  mtype* const __restrict__ kcols)
{
  __shared__ mtype temp[tile * tile];
  __shared__ mtype krow[tile * tile];

  const int warp_a = threadIdx.x / ws; // i: 0-31, upper half
  const int warp_b = warp_a + ws; // i: 32-63, lower half
  const int lane_a = threadIdx.x % ws; // j: 0-31, left half
  const int lane_b = lane_a + ws; // j: 32-63, right half

  const int idx0_aa = warp_a * upper + lane_a; // upper left
  const int idx0_ab = warp_a * upper + lane_b; // upper right
  const int idx0_ba = warp_b * upper + lane_a; // lower left
  const int idx0_bb = warp_b * upper + lane_b; // lower right

  const int idx1_aa = lane_a * tile + warp_a;
  const int idx1_ab = lane_b * tile + warp_a;
  const int idx1_ba = lane_a * tile + warp_b;
  const int idx1_bb = lane_b * tile + warp_b;

  int idx2_a = lane_a;
  int idx2_b = lane_b;

  mtype ij_aa = AdjMat[idx0_aa];
  mtype ij_ab = AdjMat[idx0_ab];
  mtype ij_ba = AdjMat[idx0_ba];
  mtype ij_bb = AdjMat[idx0_bb];

  #pragma unroll 64
  for (int k = 0; k < tile; k++) {
    if (warp_a == k) krow[idx2_a] = ij_aa;
    if (warp_a == k) krow[idx2_b] = ij_ab;
    if (warp_b == k) krow[idx2_a] = ij_ba;
    if (warp_b == k) krow[idx2_b] = ij_bb;
    __syncthreads();

    mtype ik_a, ik_b;
    if (k < ws) {
      ik_a = __shfl(ij_aa, k, ws);
      ik_b = __shfl(ij_ba, k, ws);
    } else {
      ik_a = __shfl(ij_ab, k - ws, ws);
      ik_b = __shfl(ij_bb, k - ws, ws);
    }

    const mtype kr_a = krow[idx2_a];
    const mtype kr_b = krow[idx2_b];

    ij_aa = min(ij_aa, ik_a + kr_a);
    ij_ab = min(ij_ab, ik_a + kr_b);
    ij_ba = min(ij_ba, ik_b + kr_a);
    ij_bb = min(ij_bb, ik_b + kr_b);

    if (warp_a == k) krows[idx0_aa] = ij_aa;
    if (warp_a == k) krows[idx0_ab] = ij_ab;
    if (warp_b == k) krows[idx0_ba] = ij_ba;
    if (warp_b == k) krows[idx0_bb] = ij_bb;

    if (lane_a == k) temp[idx1_aa] = ij_aa;
    if (lane_a == k) temp[idx1_ba] = ij_ba;
    if (lane_b == k) temp[idx1_ab] = ij_ab;
    if (lane_b == k) temp[idx1_bb] = ij_bb;

    idx2_a += tile;
    idx2_b += tile;
  }

  __syncthreads();
  kcols[idx0_aa] = temp[warp_a * tile + lane_a];
  kcols[idx0_ab] = temp[warp_a * tile + lane_b];
  kcols[idx0_ba] = temp[warp_b * tile + lane_a];
  kcols[idx0_bb] = temp[warp_b * tile + lane_b];
  AdjMat[idx0_aa] = ij_aa;
  AdjMat[idx0_ab] = ij_ab;
  AdjMat[idx0_ba] = ij_ba;
  AdjMat[idx0_bb] = ij_bb;
}

// calculate k^th row and col
extern "C" __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void FWrowcol_64(
  mtype* const __restrict__ AdjMat,
  const int upper,
  mtype* const __restrict__ krows,
  mtype* const __restrict__ kcols,
  const int x, const int subm1)
{
  __shared__ mtype temp[tile * tile];
  __shared__ mtype krow[tile * tile];

  const int warp_a = threadIdx.x / ws; // i: 0-31, upper half
  const int warp_b = warp_a + ws; // i: 32-63, lower half
  const int lane_a = threadIdx.x % ws; // j: 0-31, left half
  const int lane_b = lane_a + ws; // j: 32-63, right half

  int y = blockIdx.x;

  if (y < subm1) {
    if (y >= x) y++;
    const int i_a = warp_a + x * tile;
    const int i_b = warp_b + x * tile;
    const int j_a = lane_a + y * tile;
    const int j_b = lane_b + y * tile;

    const int idx0_aa = i_a * upper + j_a;
    const int idx0_ab = i_a * upper + j_b;
    const int idx0_ba = i_b * upper + j_a;
    const int idx0_bb = i_b * upper + j_b;

    int idx1_a = warp_a;
    int idx1_b = warp_b;
    int idx2_a = lane_a;
    int idx2_b = lane_b;

    temp[warp_a * tile + lane_a] = kcols[i_a * upper + lane_a + x * tile];
    temp[warp_a * tile + lane_b] = kcols[i_a * upper + lane_b + x * tile];
    temp[warp_b * tile + lane_a] = kcols[i_b * upper + lane_a + x * tile];
    temp[warp_b * tile + lane_b] = kcols[i_b * upper + lane_b + x * tile];
    __syncthreads();

    mtype ij_aa = AdjMat[idx0_aa];
    mtype ij_ab = AdjMat[idx0_ab];
    mtype ij_ba = AdjMat[idx0_ba];
    mtype ij_bb = AdjMat[idx0_bb];

    const mtype orig_aa = ij_aa;
    const mtype orig_ab = ij_ab;
    const mtype orig_ba = ij_ba;
    const mtype orig_bb = ij_bb;

    #pragma unroll 64
    for (int k = 0; k < tile; k++) {
      if (warp_a == k) krow[idx2_a] = ij_aa;
      if (warp_a == k) krow[idx2_b] = ij_ab;
      if (warp_b == k) krow[idx2_a] = ij_ba;
      if (warp_b == k) krow[idx2_b] = ij_bb;
      __syncthreads();

      const mtype ik_a = temp[idx1_a];
      const mtype ik_b = temp[idx1_b];
      const mtype kr_a = krow[idx2_a];
      const mtype kr_b = krow[idx2_b];

      ij_aa = min(ij_aa, ik_a + kr_a);
      ij_ab = min(ij_ab, ik_a + kr_b);
      ij_ba = min(ij_ba, ik_b + kr_a);
      ij_bb = min(ij_bb, ik_b + kr_b);

      if (warp_a == k) krows[idx0_aa] = ij_aa;
      if (warp_a == k) krows[idx0_ab] = ij_ab;
      if (warp_b == k) krows[idx0_ba] = ij_ba;
      if (warp_b == k) krows[idx0_bb] = ij_bb;

      idx1_a += tile;
      idx1_b += tile;
      idx2_a += tile;
      idx2_b += tile;
    }
    if (ij_aa != orig_aa) AdjMat[idx0_aa] = ij_aa;
    if (ij_ab != orig_ab) AdjMat[idx0_ab] = ij_ab;
    if (ij_ba != orig_ba) AdjMat[idx0_ba] = ij_ba;
    if (ij_bb != orig_bb) AdjMat[idx0_bb] = ij_bb;
  } else {
    y -= subm1;
    if (y >= x) y++;
    const int i_a = warp_a + y * tile;
    const int i_b = warp_b + y * tile;

    const int j_a = lane_a + x * tile;
    const int j_b = lane_b + x * tile;

    const int idx0_aa = i_a * upper + j_a;
    const int idx0_ab = i_a * upper + j_b;
    const int idx0_ba = i_b * upper + j_a;
    const int idx0_bb = i_b * upper + j_b;

    const int idx1_aa = lane_a * tile + warp_a;
    const int idx1_ab = lane_b * tile + warp_a;
    const int idx1_ba = lane_a * tile + warp_b;
    const int idx1_bb = lane_b * tile + warp_b;

    int idx2_a = (x * tile) * upper + j_a;
    int idx2_b = (x * tile) * upper + j_b;

    mtype ij_aa = AdjMat[idx0_aa];
    mtype ij_ab = AdjMat[idx0_ab];
    mtype ij_ba = AdjMat[idx0_ba];
    mtype ij_bb = AdjMat[idx0_bb];

    const mtype orig_aa = ij_aa;
    const mtype orig_ab = ij_ab;
    const mtype orig_ba = ij_ba;
    const mtype orig_bb = ij_bb;

    #pragma unroll 64
    for (int k = 0; k < tile; k++) {
      mtype ik_a, ik_b;
      if (k < ws) {
        ik_a = __shfl(ij_aa, k, ws);
        ik_b = __shfl(ij_ba, k, ws);
      }
      else {
        ik_a = __shfl(ij_ab, k - ws, ws);
        ik_b = __shfl(ij_bb, k - ws, ws);
      }
      const mtype kr_a = krows[idx2_a];
      const mtype kr_b = krows[idx2_b];

      ij_aa = min(ij_aa, ik_a + kr_a);
      ij_ab = min(ij_ab, ik_a + kr_b);
      ij_ba = min(ij_ba, ik_b + kr_a);
      ij_bb = min(ij_bb, ik_b + kr_b);

      if (lane_a == k) temp[idx1_aa] = ij_aa;
      if (lane_a == k) temp[idx1_ba] = ij_ba;
      if (lane_b == k) temp[idx1_ab] = ij_ab;
      if (lane_b == k) temp[idx1_bb] = ij_bb;

      idx2_a += upper;
      idx2_b += upper;
    }
    __syncthreads();

    kcols[idx0_aa] = temp[warp_a * tile + lane_a];
    kcols[idx0_ab] = temp[warp_a * tile + lane_b];
    kcols[idx0_ba] = temp[warp_b * tile + lane_a];
    kcols[idx0_bb] = temp[warp_b * tile + lane_b];

    if (ij_aa != orig_aa) AdjMat[idx0_aa] = ij_aa;
    if (ij_ab != orig_ab) AdjMat[idx0_ab] = ij_ab;
    if (ij_ba != orig_ba) AdjMat[idx0_ba] = ij_ba;
    if (ij_bb != orig_bb) AdjMat[idx0_bb] = ij_bb;
  }
}

// calculate the remaining tiles
extern "C" __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void FWrem_64(
  mtype* const __restrict__ AdjMat,
  const int upper,
  mtype* const __restrict__ krows,
  mtype* const __restrict__ kcols,
  const int x, const int subm1)
{
  int y = blockIdx.x / subm1;
  int z = blockIdx.x % subm1;
  if (y >= x) y++;
  if (z >= x) z++;

  const int warp_a = threadIdx.x / ws;
  const int warp_b = warp_a + ws;

  const int lane_a = threadIdx.x % ws;
  const int lane_b = lane_a + ws;

  const int i_a = warp_a + y * tile;
  const int i_b = warp_b + y * tile;
  const int j_a = lane_a + z * tile;
  const int j_b = lane_b + z * tile;

  const int idx0_aa = i_a * upper + j_a; // upper left
  const int idx0_ab = i_a * upper + j_b; // upper right
  const int idx0_ba = i_b * upper + j_a; // lower left
  const int idx0_bb = i_b * upper + j_b; // lower right

  __shared__ mtype s_kj[tile * tile];
  __shared__ mtype s_ik[tile * tile];

  s_kj[warp_a * tile + lane_a] = krows[(x * tile + warp_a) * upper + j_a];
  s_kj[warp_a * tile + lane_b] = krows[(x * tile + warp_a) * upper + j_b];
  s_kj[warp_b * tile + lane_a] = krows[(x * tile + warp_b) * upper + j_a];
  s_kj[warp_b * tile + lane_b] = krows[(x * tile + warp_b) * upper + j_b];

  s_ik[warp_a * tile + lane_a] = kcols[i_a * upper + lane_a + x * tile];
  s_ik[warp_a * tile + lane_b] = kcols[i_a * upper + lane_b + x * tile];
  s_ik[warp_b * tile + lane_a] = kcols[i_b * upper + lane_a + x * tile];
  s_ik[warp_b * tile + lane_b] = kcols[i_b * upper + lane_b + x * tile];

  mtype ij_aa = AdjMat[idx0_aa];
  mtype ij_ab = AdjMat[idx0_ab];
  mtype ij_ba = AdjMat[idx0_ba];
  mtype ij_bb = AdjMat[idx0_bb];

  const mtype orig_aa = ij_aa;
  const mtype orig_ab = ij_ab;
  const mtype orig_ba = ij_ba;
  const mtype orig_bb = ij_bb;

  __syncthreads();
  int idx1_a = warp_a;
  int idx1_b = warp_b;

  int idx2_a = lane_a;
  int idx2_b = lane_b;

  #pragma unroll 64
  for (int k = 0; k < tile; k++) {
    const mtype sk_a = s_kj[idx2_a];
    const mtype sk_b = s_kj[idx2_b];

    ij_aa = min(ij_aa, s_ik[idx1_a] + sk_a);
    ij_ab = min(ij_ab, s_ik[idx1_a] + sk_b);
    ij_ba = min(ij_ba, s_ik[idx1_b] + sk_a);
    ij_bb = min(ij_bb, s_ik[idx1_b] + sk_b);

    idx1_a += tile;
    idx1_b += tile;

    idx2_a += tile;
    idx2_b += tile;
  }

  __syncthreads();

  if ((y == z) && (y == x + 1) && (x != subm1)) { // the diagonal in next iteration
    const int idx1_aa = lane_a * tile + warp_a;
    const int idx1_ab = lane_b * tile + warp_a;
    const int idx1_ba = lane_a * tile + warp_b;
    const int idx1_bb = lane_b * tile + warp_b;

    int idx2_a = lane_a;
    int idx2_b = lane_b;

    #pragma unroll 64
    for (int k = 0; k < tile; k++) {
      if (warp_a == k) s_kj[idx2_a] = ij_aa;
      if (warp_a == k) s_kj[idx2_b] = ij_ab;
      if (warp_b == k) s_kj[idx2_a] = ij_ba;
      if (warp_b == k) s_kj[idx2_b] = ij_bb;

      mtype ik_a, ik_b;
      if (k < ws) {
        ik_a = __shfl(ij_aa, k, ws);
        ik_b = __shfl(ij_ba, k, ws);
      }
      else {
        ik_a = __shfl(ij_ab, k - ws, ws);
        ik_b = __shfl(ij_bb, k - ws, ws);
      }

      __syncthreads();

      const mtype sk_a = s_kj[idx2_a];
      const mtype sk_b = s_kj[idx2_b];

      ij_aa = min(ij_aa, ik_a + sk_a);
      ij_ab = min(ij_ab, ik_a + sk_b);
      ij_ba = min(ij_ba, ik_b + sk_a);
      ij_bb = min(ij_bb, ik_b + sk_b);

      if (warp_a == k) krows[idx0_aa] = ij_aa;
      if (warp_a == k) krows[idx0_ab] = ij_ab;
      if (warp_b == k) krows[idx0_ba] = ij_ba;
      if (warp_b == k) krows[idx0_bb] = ij_bb;

      if (lane_a == k) s_ik[idx1_aa] = ij_aa;
      if (lane_a == k) s_ik[idx1_ba] = ij_ba;
      if (lane_b == k) s_ik[idx1_ab] = ij_ab;
      if (lane_b == k) s_ik[idx1_bb] = ij_bb;
      idx2_a += tile;
      idx2_b += tile;
    }
    __syncthreads();

    kcols[idx0_aa] = s_ik[warp_a * tile + lane_a];
    kcols[idx0_ab] = s_ik[warp_a * tile + lane_b];
    kcols[idx0_ba] = s_ik[warp_b * tile + lane_a];
    kcols[idx0_bb] = s_ik[warp_b * tile + lane_b];
  }

  if (ij_aa != orig_aa) AdjMat[idx0_aa] = ij_aa;
  if (ij_ab != orig_ab) AdjMat[idx0_ab] = ij_ab;
  if (ij_ba != orig_ba) AdjMat[idx0_ba] = ij_ba;
  if (ij_bb != orig_bb) AdjMat[idx0_bb] = ij_bb;
}
