/*
   ECL-MIS code: ECL-MIS is a maximal independent set algorithm that is quite
   fast and produces relatively large sets. It operates on graphs stored in
   binary CSR format.

   Copyright (c) 2017-2020, Texas State University. All rights reserved. Patented.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted (subject to the limitations in the disclaimer
   below) provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY
 BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher and Sindhu Devale

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-MIS/.

Publication: This work is described in detail in the following paper.
Martin Burtscher, Sindhu Devale, Sahar Azimi, Jayadharini Jaiganesh, and
Evan Powers. A High-Quality and Fast Maximal Independent Set Implementation
for GPUs. ACM Transactions on Parallel Computing, Vol. 5, No. 2, Article 8
(27 pages). December 2018.
 */


#include <hip/hip_runtime.h>
static const int ThreadsPerBlock=256;
using stattype=unsigned char;
static const stattype in=0xfe,out=0;
extern "C" __global__ 
void findmins(const int nodes, 
    const int* const __restrict nidx,
    const int* const __restrict nlist,
    volatile stattype* const __restrict nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  int missing;
  do {
    missing = 0;
    for (int v = from; v < nodes; v += incr) {
      const stattype nv = nstat[v];
      if (nv & 1) {
        int i = nidx[v];
        while ((i < nidx[v + 1]) && ((nv > nstat[nlist[i]]) || ((nv == nstat[nlist[i]]) && (v > nlist[i])))) {
          i++;
        }
        if (i < nidx[v + 1]) {
          missing = 1;
        } else {
          for (int i = nidx[v]; i < nidx[v + 1]; i++) {
            nstat[nlist[i]] = out;
          }
          nstat[v] = in;
        }
      }
    }
  } while (missing != 0);
}

/* hash function to generate random values */

// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
__device__ 
unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

/* prioritized-selection initialization kernel */

extern "C" __global__
void init(const int nodes, 
    const int edges, 
    const int* const __restrict nidx,
    stattype* const __restrict nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  const float avg = (float)edges / nodes;
  const float scaledavg = ((in / 2) - 1) * avg;

  for (int i = from; i < nodes; i += incr) {
    stattype val = in;
    const int degree = nidx[i + 1] - nidx[i];
    if (degree > 0) {
      float x = degree - (hash(i) * 0.00000000023283064365386962890625f);
      int res = int(scaledavg / (avg + x));
      val = (res + res) | 1;
    }
    nstat[i] = val;
  }
}
