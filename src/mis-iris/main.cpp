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


#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "iris_dmem.hpp"
#include "graph.h"

static const int ThreadsPerBlock = 256;

typedef unsigned char stattype;
static const stattype in = 0xfe;
static const stattype out = 0;

/* main computation kernel */

void computeMIS(
    const int repeat,
    const int nodes,
    const int edges,
    const int* const __restrict nidx,
    const int* const __restrict nlist,
    stattype* const __restrict nstat)
{
  int* nidx_d;
  int* nlist_d;
  stattype* nstat_d;

  if (IRIS_SUCCESS != iris_alloc((void **)&nidx_d, (nodes + 1) * sizeof(int))) {
    fprintf(stderr, "ERROR: could not allocate nidx_d\n\n");
  }
  if (IRIS_SUCCESS != iris_alloc((void **)&nlist_d, edges * sizeof(int))) {
    fprintf(stderr, "ERROR: could not allocate nlist_d\n\n");
  }
  if (IRIS_SUCCESS != iris_alloc((void **)&nstat_d, nodes * sizeof(stattype))) {
    fprintf(stderr, "ERROR: could not allocate nstat_d\n\n");
  }
  if (IRIS_SUCCESS != iris_copy(nidx_d, nidx, (nodes + 1) * sizeof(int), IRISHostToDevice)) {
    fprintf(stderr, "ERROR: copying to device failed\n\n");
  }
  if (IRIS_SUCCESS != iris_copy(nlist_d, nlist, edges * sizeof(int), IRISHostToDevice)) {
    fprintf(stderr, "ERROR: copying to device failed\n\n");
  }

  const int blocks = 24;

  iris_sync();

  auto start = std::chrono::high_resolution_clock::now();

  for (int n = 0; n < repeat; n++) {
    iris_launch("init", blocks, ThreadsPerBlock, nodes, edges, nidx_d, nstat_d);

    iris_launch("findmins", blocks, ThreadsPerBlock, nodes, nidx_d, nlist_d, nstat_d);
  }

  iris_sync();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  float runtime = (float)elapsed_seconds.count() / repeat;
  printf("compute time: %.6f s\n", runtime);
  printf("throughput: %.6f Mnodes/s\n", nodes * 0.000001 / runtime);
  printf("throughput: %.6f Medges/s\n", edges * 0.000001 / runtime);

  if (IRIS_SUCCESS != iris_copy(nstat, nstat_d, nodes * sizeof(stattype), IRISDeviceToHost)) {
    fprintf(stderr, "ERROR: copying from device failed\n\n");
  }

  iris_free(nstat_d);
  iris_free(nlist_d);
  iris_free(nidx_d);
}

int main(int argc, char* argv[])
{
  printf("ECL-MIS v1.3 (%s)\n", __FILE__);
  printf("Copyright 2017-2020 Texas State University\n");

  if (argc != 3) {
    fprintf(stderr, "USAGE: %s <input_file_name> <repeat>\n\n", argv[0]);
    exit(-1);
  }

  IRISSession session(&argc,&argv);
  ECLgraph g = readECLgraph(argv[1]);
  printf("configuration: %d nodes and %d edges (%s)\n", g.nodes, g.edges, argv[1]);
  printf("average degree: %.2f edges per node\n", 1.0 * g.edges / g.nodes);

  stattype* nstatus = (stattype*)malloc(g.nodes * sizeof(nstatus[0]));

  if (nstatus == NULL) {
    fprintf(stderr, "ERROR: could not allocate nstatus\n\n");
  }
  else {
    const int repeat = atoi(argv[2]);

    computeMIS(repeat, g.nodes, g.edges, g.nindex, g.nlist, nstatus);

    /* result verification code */

    for (int v = 0; v < g.nodes; v++) {
      if ((nstatus[v] != in) && (nstatus[v] != out)) {
        fprintf(stderr, "ERROR: found unprocessed node in graph\n\n");
        break;
      }
      if (nstatus[v] == in) {
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
          if (nstatus[g.nlist[i]] == in) {
            fprintf(stderr, "ERROR: found adjacent nodes in MIS\n\n");
            break;
          }
        }
      } else {
        int flag = 0;
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
          if (nstatus[g.nlist[i]] == in) {
            flag = 1;
          }
        }
        if (flag == 0) {
          fprintf(stderr, "ERROR: set is not maximal\n\n");
          break;
        }
      }
    }
  }

  bool verified = nstatus != nullptr;
  unsigned long long checksum=1469598103934665603ULL;
  if(nstatus) for(int v=0;v<g.nodes;++v) {
    checksum=(checksum ^ nstatus[v])*1099511628211ULL;
    if(nstatus[v]!=in && nstatus[v]!=out) verified=false;
    bool neighbor=false;
    for(int e=g.nindex[v];e<g.nindex[v+1];++e) {
      if(nstatus[g.nlist[e]]==in) neighbor=true;
    }
    if(nstatus[v]==in ? neighbor : !neighbor) verified=false;
  }
  printf("%s\n",verified?"PASS":"FAIL");
  printf("Checksum: %llu\n",checksum);
  freeECLgraph(g);
  if (nstatus != NULL) free(nstatus);
  return verified?0:1;
}
