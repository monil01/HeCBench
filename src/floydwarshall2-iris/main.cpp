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


#include "dmem.hpp"
static void FW_gpu_64(const ECLgraph g, mtype* const AdjMat, const int repeat)
{
  // copy graph to GPU
  ECLgraph d_g = g;
  port::allocate((void **)&d_g.nindex, sizeof(int) * (g.nodes + 1));
  port::allocate((void **)&d_g.nlist, sizeof(int) * g.edges);
  port::allocate((void **)&d_g.eweight, sizeof(int) * g.edges);
  port::copy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), port::h2d);
  port::copy(d_g.nlist, g.nlist, sizeof(int) * g.edges, port::h2d);
  port::copy(d_g.eweight, g.eweight, sizeof(int) * g.edges, port::h2d);

  // allocate GPU memory
  const int sub = (g.nodes + tile - 1) / tile;

  const int upper = sub * tile; // upper bound of the GPU matrix
  mtype* d_AdjMat;
  if (hipSuccess != port::allocate((void **)&d_AdjMat, sizeof(mtype) * upper * upper))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  mtype* d_krows;
  if (hipSuccess != port::allocate((void **)&d_krows, sizeof(mtype) * upper * upper))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  mtype* d_kcols;
  if (hipSuccess != port::allocate((void **)&d_kcols, sizeof(mtype) * upper * upper))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  printf("GPU matrix size: %.1f MB\n", sizeof(mtype) * upper * upper / (1024.0 * 1024.0));

  iris_synchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // run GPU init code
    port::launch("init1",dim3((upper * upper + ThreadsPerBlock - 1) / ThreadsPerBlock),dim3(ThreadsPerBlock),g.nodes, d_AdjMat, upper);
    port::launch("init2",dim3((g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock),dim3(ThreadsPerBlock),g.nodes,d_g.nindex,d_g.nlist,d_g.eweight,d_AdjMat,upper);
  }

  iris_synchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (initialization) time: %10.6f s\n", time * 1e-9 / repeat);

  const int subm1 = sub - 1;
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // compute 64*64 tile
    port::launch("FW0_64",dim3(1),dim3(ThreadsPerBlock),d_AdjMat, upper, d_krows, d_kcols);

    if (sub > 1) {
      for (int x = 0; x < sub; x++) {
        port::launch("FWrowcol_64",dim3(2 * subm1),dim3(ThreadsPerBlock),d_AdjMat, upper, d_krows, d_kcols, x, subm1);
        port::launch("FWrem_64",dim3(subm1 * subm1),dim3(ThreadsPerBlock),d_AdjMat, upper, d_krows, d_kcols, x, subm1);
      }
    }
  }
  iris_synchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (compute) time: %10.6f s\n", time * 1e-9 / repeat);

  // copy result back to CPU
  if (hipSuccess != port::copy(AdjMat, d_AdjMat, sizeof(mtype) * upper * upper, port::d2h))
    fprintf(stderr, "ERROR: copying from device failed\n");

  // clean up
  port::release(d_g.nindex);
  port::release(d_g.nlist);
  port::release(d_g.eweight);
  port::release(d_AdjMat);
  port::release(d_krows);
  port::release(d_kcols);
}

static void FW_cpu(const ECLgraph g, mtype* const AdjMat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < g.nodes; i++) {
    for (int j = 0; j < g.nodes; j++) {
      AdjMat[i * g.nodes + j] = ((i == j) ? 0 : (INT_MAX / 2));
    }
  }

  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      AdjMat[i * g.nodes + nei] = g.eweight[j];
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("CPU init time: %10.6f s\n", time * 1e-9);

  start = std::chrono::steady_clock::now();

  for (int k = 0; k < g.nodes; k++) {
    for (int i = 0; i < g.nodes; i++) {
      for (int j = 0; j < g.nodes; j++) {
        if (AdjMat[i * g.nodes + j] > AdjMat[i * g.nodes + k] + AdjMat[k * g.nodes + j]) {
          AdjMat[i * g.nodes + j] = AdjMat[i * g.nodes + k] + AdjMat[k * g.nodes + j];
        }
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("CPU comp time: %10.6f s\n", time * 1e-9);
}

int main(int argc, char* argv[])
{
  port::Session session(&argc, &argv);
  printf("ECL-APSP v1.0 (%s)\n", __FILE__);
  printf("Copyright 2021 Texas State University\n");
  if (argc != 3) {
    fprintf(stderr, "USAGE: %s <input_graph_name> <repeat>\n\n", argv[0]);
    return 1;
  }
  if (ThreadsPerBlock != 1024) {
    fprintf(stderr, "Threads per block must be 1024\n\n");
    return 1;
  }

  // allocation of matrices may fail on a host
  mtype* AdjMat1 = NULL;
  mtype* AdjMat2 = NULL;

  // declare them before the goto statement
  int upper_64;
  int diffcount;
  int gn;

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  const int repeat = atoi(argv[2]);

  if (g.eweight == NULL) {
    fprintf(stderr, "ERROR: input graph has no edge weights\n\n");
    goto DONE;
  }

  // make all weights positive to avoid negative cycles
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
    }
  }

  // run on device
  upper_64 = ((g.nodes + tile - 1) / tile) * tile;  // round up
  AdjMat1 = (mtype*) malloc (sizeof(mtype) * upper_64 * upper_64);
  if (AdjMat1 == NULL) {
    fprintf(stderr, "ERROR: memory allocation (AdjMat1) fails\n\n");
    goto DONE;
  }

  FW_gpu_64(g, AdjMat1, repeat);

  // run on host
  AdjMat2 = (mtype*) malloc (sizeof(mtype) * g.nodes * g.nodes);
  if (AdjMat2 == NULL) {
    fprintf(stderr, "ERROR: memory allocation (AdjMat2) fails\n\n");
    goto DONE;
  }

  FW_cpu(g, AdjMat2);

  // compare results
  diffcount = 0;
  gn = g.nodes;
  for (int i = 0; i < gn; ++i) {
    for (int j = 0; j < gn; ++j) {
      if (AdjMat1[i * upper_64 + j] != AdjMat2[i * g.nodes + j]) {
        diffcount++;
      }
    }
  }

  if (diffcount > 0) {
    printf("ERROR: results differ in %d places!\n", diffcount);
  } else {
    printf("results match\n");
  }

  DONE:
  // clean up
  if (AdjMat1) free(AdjMat1);
  if (AdjMat2) free(AdjMat2);
  freeECLgraph(g);
  return 0;
}
