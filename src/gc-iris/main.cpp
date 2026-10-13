/*
   ECL-GC code: ECL-GC is a graph-coloring algorithm with shortcutting. The CUDA
   implementation thereof is quite fast. It operates on graphs stored in binary
   CSR format.

   Copyright (c) 2020, Texas State University. All rights reserved.

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

Authors: Ghadeer Alabandi, Evan Powers, and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-GC/.

Publication: This work is described in detail in the following paper.
Ghadeer Alabandi, Evan Powers, and Martin Burtscher. Increasing the Parallelism
of Graph Coloring via Shortcutting. Proceedings of the 2020 ACM SIGPLAN
Symposium on Principles and Practice of Parallel Programming, pp. 262-275.
February 2020.
 */


#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include "dmem_runtime.hpp"
#include "graph.h"
#include "dispatch.h"

static const int ThreadsPerBlock = 512;
static const int WS = 32;  // word size and bits per int
static const int MSB = 1 << (WS - 1);
static const int Mask = (1 << (WS / 2)) - 1;


// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
int main(int argc, char* argv[])
{
  printf("ECL-GC v1.2 (%s)\n", __FILE__);
  printf("Copyright 2020 Texas State University\n\n");

  if (argc != 3) {printf("USAGE: %s <input_file_name> <repeat>\n\n", argv[0]);  exit(-1);}

  int Warp_Size;
  hipDeviceGetAttribute(&Warp_Size, hipDeviceAttributeWarpSize, 0);
  if (WS != sizeof(int) * 8) {printf("ERROR: bits per word must match warp size\n\n");  exit(-1);}
  if ((ThreadsPerBlock < Warp_Size) || ((ThreadsPerBlock % Warp_Size) != 0)) {
    printf("ERROR: threads per block must be a multiple of the warp size\n\n");
    exit(-1);
  }
  if ((ThreadsPerBlock & (ThreadsPerBlock - 1)) != 0) {
    printf("ERROR: threads per block must be a power of two\n\n");
    exit(-1);
  }

  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  const int repeat = atoi(argv[2]);

  int* const color = new int [g.nodes];

  int *nidx_d, *nlist_d, *nlist2_d, *posscol_d, *posscol2_d, *color_d, *wl_d, *wlsize_d;
  if (IRIS_SUCCESS != iris_alloc((void **)&nidx_d, (g.nodes + 1) * sizeof(int))) 
    printf("ERROR: could not allocate nidx_d\n\n");
  if (IRIS_SUCCESS != iris_alloc((void **)&nlist_d, g.edges * sizeof(int)))
    printf("ERROR: could not allocate nlist_d\n\n");
  if (IRIS_SUCCESS != iris_alloc((void **)&nlist2_d, g.edges * sizeof(int)))
    printf("ERROR: could not allocate nlist2_d\n\n");
  if (IRIS_SUCCESS != iris_alloc((void **)&posscol_d, g.nodes * sizeof(int))) 
    printf("ERROR: could not allocate posscol_d\n\n");
  if (IRIS_SUCCESS != iris_alloc((void **)&posscol2_d, (g.edges / WS + 1) * sizeof(int))) 
    printf("ERROR: could not allocate posscol2_d\n\n");
  if (IRIS_SUCCESS != iris_alloc((void **)&color_d, g.nodes * sizeof(int))) 
    printf("ERROR: could not allocate color_d\n\n");
  if (IRIS_SUCCESS != iris_alloc((void **)&wl_d, g.nodes * sizeof(int))) 
    printf("ERROR: could not allocate wl_d\n\n");
  if (IRIS_SUCCESS != iris_alloc((void **)&wlsize_d, sizeof(int))) 
    printf("ERROR: could not allocate wlsize\n\n");

  if (IRIS_SUCCESS != iris_copy(nidx_d, g.nindex, (g.nodes + 1) * sizeof(int), iris_h2d)) 
    printf("ERROR: copying nidx to device failed\n\n");
  if (IRIS_SUCCESS != iris_copy(nlist_d, g.nlist, g.edges * sizeof(int), iris_h2d)) 
    printf("ERROR: copying nlist to device failed\n\n");

  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * mTpSM / ThreadsPerBlock;
  printf("Total number of compute units: %d\n", SMs);
  printf("Maximum resident threads per compute unit: %d\n", mTpSM);
  printf("Work-group size: %d\n", ThreadsPerBlock);
  printf("Total number of work-groups: %d\n", blocks);

  iris_sync();

  auto start = std::chrono::high_resolution_clock::now();

  for (int n = 0; n < repeat; n++) {
    iris_fill(wlsize_d, 0, sizeof(int));
    dispatch_init( dim3(blocks), dim3(ThreadsPerBlock), g.nodes, g.edges, nidx_d, nlist_d, nlist2_d, posscol_d, posscol2_d, color_d, wl_d, wlsize_d);
    dispatch_runLarge( dim3(blocks), dim3(ThreadsPerBlock), g.nodes, nidx_d, nlist2_d, posscol_d, posscol2_d, color_d, wl_d, wlsize_d);
    dispatch_runSmall( dim3(blocks), dim3(ThreadsPerBlock), g.nodes, nidx_d, nlist_d, posscol_d, color_d);
  }

  iris_sync();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  float runtime = elapsed_seconds.count() / repeat;

  printf("average runtime (%d runs):    %.6f s\n", repeat, runtime);
  printf("throughput: %.6f Mnodes/s\n", g.nodes * 0.000001 / runtime);
  printf("throughput: %.6f Medges/s\n", g.edges * 0.000001 / runtime);

  if (IRIS_SUCCESS != iris_copy(color, color_d, g.nodes * sizeof(int), iris_d2h)) 
    printf("ERROR: copying color from device failed\n\n");

  iris_free(wlsize_d);
  iris_free(wl_d);
  iris_free(color_d);
  iris_free(posscol2_d);
  iris_free(posscol_d);
  iris_free(nlist2_d);
  iris_free(nlist_d);
  iris_free(nidx_d);

  bool ok = true;
  for (int v = 0; v < g.nodes; v++) {
    if (color[v] < 0) {
      printf("ERROR: found unprocessed node in graph (node %d with deg %d)\n\n",
             v, g.nindex[v + 1] - g.nindex[v]);
      ok = false;
      break;
    }
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      if (color[g.nlist[i]] == color[v]) {
        printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n",
               color[v], v, g.nlist[i]);
        ok = false;
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  if (ok) {
    const int vals = 16;
    int c[vals];
    for (int i = 0; i < vals; i++) c[i] = 0;
    int cols = -1;
    for (int v = 0; v < g.nodes; v++) {
      cols = std::max(cols, color[v]);
      if (color[v] < vals) c[color[v]]++;
    }
    cols++;
    printf("Number of distinct colors used: %d\n", cols);

    int sum = 0;
    for (int i = 0; i < std::min(vals, cols); i++) {
      sum += c[i];
      printf("color %2d: %10d (%5.1f%%)\n", i, c[i], 100.0 * sum / g.nodes);
    }
  }

  delete [] color;
  freeECLgraph(g);
  iris_finish();
  return 0;
}
