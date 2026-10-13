/*
graphB+ balancing algorithm for signed social network graphs

Copyright 2021, Texas State University. All rights reserved.

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

Authors: Ghadeer Alabandi and Martin Burtscher

URL: The latest version of this code is available at
https://cs.txstate.edu/~burtscher/research/graphB/.
*/


#include <cstdio>
#include <chrono>
#include <climits>
#include <algorithm>
#include <set>
#include <map>
#include <hip/hip_runtime.h>
#include <iris/iris.hpp>
#include <vector>
#include "graph.h"

int main(int argc, char* argv[])
{
  printf("graphB+ balancing code for signed social network graphs (%s)\n", __FILE__);
  printf("Copyright 2021 Texas State University\n");
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 1;
  int backend=0,type=0;char name[256]={};
  iris_device_info(Device,iris_backend,&backend,nullptr);iris_device_info(Device,iris_type,&type,nullptr);
  iris_device_info(Device,iris_name,name,nullptr);
  printf("Device backend=%d type=%d name=%s\n",backend,type,name);
  if(backend!=3 || type!=256) return 1;

  // process command line and read input
  if (argc != 4) {printf("USAGE: %s input_file_name iteration_count output_file_name\n", argv[0]); exit(-1);}

#ifdef VERIFY
  printf("verification is on\n");
#else
  printf("verification is off\n");
#endif
  printf("input: %s\n", argv[1]);
  Graph g = readGraph(argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int iterations = atoi(argv[2]);

  // allocate all memory
  bool* const minus = new bool [g.edges];
  int* const parent = new int [g.nodes];
  int* const queue = new int [g.nodes];  // first used as queue, then as CC size
  int* const label = new int [g.nodes];  // first used as count, then as label, and finally as CC label
  int* const border = new int [g.nodes + 2];  // maybe make smaller
  int* const inCC = new int [g.nodes];  // how often node was in largest CC or at an even distance from largest CC
  int* const inTree = new int [g.edges];  // how often edge was in tree
  int* const negCnt = new int [g.edges];  // how often edge was negative
  EdgeInfo* const einfo = new EdgeInfo [g.edges + 16];
  int* const root = new int [g.nodes];  // tree roots

  for (int i = 0; i < g.nodes; i++) root[i] = i;
  std::partial_sort(root, root + std::min(iterations, g.nodes), root + g.nodes, [&](int a, int b) {
    return (g.nindex[a + 1] - g.nindex[a]) > (g.nindex[b + 1] - g.nindex[b]);
  });

  //GPU code
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, Device);
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  int clockRate, memoryClockRate;
  hipDeviceGetAttribute(&clockRate, hipDeviceAttributeClockRate, 0);
  hipDeviceGetAttribute(&memoryClockRate, hipDeviceAttributeMemoryClockRate, 0);
  printf("Device: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n",
         deviceProp.name, SMs, mTpSM, clockRate * 0.001, memoryClockRate * 0.001);

  int host_tail=0,host_wsize=0;bool host_changed=false;
  std::vector<int> ws1(g.edges),ws2(g.edges);
  iris::DMem d_eweight(g.eweight,sizeof(int)*g.edges);
  iris::DMem d_nindex(g.nindex,sizeof(int)*(g.nodes+1));
  iris::DMem d_nlist(g.nlist,sizeof(int)*g.edges);
  iris::DMem d_inTree(inTree,sizeof(int)*g.edges);
  iris::DMem d_negCnt(negCnt,sizeof(int)*g.edges);
  iris::DMem d_inCC(inCC,sizeof(int)*g.nodes);
  iris::DMem d_einfo(einfo,sizeof(EdgeInfo)*g.edges);
  iris::DMem d_label(label,sizeof(int)*g.nodes);
  iris::DMem d_parent(parent,sizeof(int)*g.nodes);
  iris::DMem d_queue(queue,sizeof(int)*g.nodes);
  iris::DMem d_border(border,sizeof(int)*(g.nodes+2));
  iris::DMem d_tail(&host_tail,sizeof(int)*1);
  iris::DMem d_ws1(ws1.data(),sizeof(int)*g.edges);
  iris::DMem d_ws2(ws2.data(),sizeof(int)*g.edges);
  iris::DMem d_wSize(&host_wsize,sizeof(int)*1);
  iris::DMem d_minus(minus,sizeof(bool)*g.edges);
  iris::DMem changed_gpu(&host_changed,sizeof(bool)*1);
  const int blocks = SMs * mTpSM / ThreadsPerBlock;

  // use random pluses and minuses
  launch("init", blocks, ThreadsPerBlock, g.edges, g.nodes, d_nlist, d_eweight, d_inCC, d_einfo, d_inTree, d_negCnt);



  int min_d = INT_MAX;
  int max_d = INT_MIN;
  int sum_d = 0;
  double avg_d = 0;

  auto start = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < iterations; iter++) {
    // generate tree
    launch("init2", blocks, ThreadsPerBlock, g.edges, g.nodes, root[iter % g.nodes], d_nlist, d_parent, d_queue, d_label, d_tail);
    int level = 0;
    int tail = 1;
    border[0] = 0;
    border[1] = tail;
    while (border[level + 1] < g.nodes) {
      launch("generateSpanningTree", blocks, ThreadsPerBlock, g.nodes, d_nindex, d_nlist, iter + 17, d_einfo, d_parent, d_queue, level, d_tail, border[level],  border[level + 1]);
      if (d_tail.fetch(&tail)!=IRIS_SUCCESS) 
        fprintf(stderr, "ERROR: copying to host failed \n");
      level++;
      border[level + 1] = tail;
    }

    const int levels = level + 1;

    //min , max and avg depth of the trees
    sum_d += level;
    if (level < min_d) min_d = level;
    if (level > max_d) max_d = level;

#ifdef VERIFY
    launch("verify_generateSpanningTree", blocks, ThreadsPerBlock, g.nodes, g.edges, d_nindex, d_nlist, iter, d_parent,
                                                             level, d_tail, border[level + 1]);
#endif
    //root count
    //#1

    for (int level = levels - 1; level > 0; level--) {
      launch("rootcount", blocks, ThreadsPerBlock, d_parent, d_queue, d_label, level, border[level],  border[level + 1]);
    }

#ifdef VERIFY
    if(d_label.fetch(label)!=IRIS_SUCCESS)
    fprintf(stderr, "ERROR: copying to host failed\n");

    if (label[root[iter % g.nodes]] != g.nodes)
    printf("ERROR: root count mismatch\n");
#endif

    // tree label
    if(d_label.fetch(label)!=IRIS_SUCCESS) return 1;
    label[root[iter % g.nodes]]=0;
    if(d_label.update(label)!=IRIS_SUCCESS) return 1;

    //#2
    for (int level = 0; level < levels; level++) {
      launch("treelabel", blocks, ThreadsPerBlock, g.nodes, d_nindex, d_nlist, d_einfo, d_inTree, d_negCnt, d_parent, 
        d_queue, d_label, level, border[level], border[level + 1]);
    }

    //#3
    launch("inTreeUpdate", blocks, ThreadsPerBlock, g.edges, d_nlist, d_inTree);
    launch("initMinus", blocks, ThreadsPerBlock, g.edges, g.nodes, d_nindex, d_nlist, d_einfo, d_minus);

    //#4
    launch("processCycles", blocks, ThreadsPerBlock, g.nodes, d_nindex, d_nlist, d_label, d_einfo, d_minus);

    launch("init3", blocks, ThreadsPerBlock, g.nodes, d_nindex, d_nlist, d_label, d_queue);
    launch("compute1", blocks, ThreadsPerBlock, g.nodes, d_nindex, d_nlist, d_label, d_minus, d_negCnt);
    launch("flatten", blocks, ThreadsPerBlock, g.nodes, d_label);
    launch("ccSize", blocks, ThreadsPerBlock, g.nodes, d_label, d_queue);
    launch("largestCC", blocks, ThreadsPerBlock, g.nodes, d_queue);
    launch("ccHopCount", blocks, ThreadsPerBlock, g.nodes, d_nindex, d_nlist, d_label, d_queue, d_ws1, d_ws2);

    bool changed;
    do {
      changed = false;
      if(changed_gpu.update(&changed)!=IRIS_SUCCESS)
        fprintf(stderr, "ERROR: setting changed failed\n");

      launch("BellmanFord", blocks, ThreadsPerBlock, d_queue, changed_gpu, d_ws1, d_ws2);
      launch("BellmanFord", blocks, ThreadsPerBlock, d_queue, changed_gpu, d_ws1, d_ws2);
      launch("BellmanFord", blocks, ThreadsPerBlock, d_queue, changed_gpu, d_ws1, d_ws2);

      if(changed_gpu.fetch(&changed)!=IRIS_SUCCESS)
        fprintf(stderr, "ERROR: copying of changed from device failed\n");
    } while (changed);

    launch("incrementCC", blocks, ThreadsPerBlock, g.nodes, d_label, d_queue, d_inCC);
  }


  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  float runtime = elapsed_seconds.count();
  printf("Total graphB+ runtime:    %.6f s\n", runtime);

  if(d_inCC.fetch(inCC)!=IRIS_SUCCESS)
  fprintf(stderr, "ERROR: copying incc from device failed\n");

  if(d_inTree.fetch(inTree)!=IRIS_SUCCESS || d_negCnt.fetch(negCnt)!=IRIS_SUCCESS ||
     d_nlist.fetch(g.nlist)!=IRIS_SUCCESS) return 1;
  // print results
  avg_d = sum_d / iterations;
  printf("number of trees %d\n", iterations);
  printf("Min depth of the trees %d\n Max depth of the trees %d\n Avg depth of the trees %.4f\n", min_d, max_d, avg_d);
  for (int i = 0; i < g.nodes; i++)
  {
    if (i >= 10)
    break; // to limit output
    printf("%6d: %6d   (%5.1f%%)  %d\n", i, inCC[i], 100.0 * inCC[i] / iterations, g.origID[i]);
  }

  // output results to file
  FILE *f = fopen(argv[3], "wt");
  fprintf(f, "original node ID, percentage node was in agreeable majority\n");
  for (int i = 0; i < g.nodes; i++) {
    fprintf(f, "%d,%.1f\n", g.origID[i], 100.0 * inCC[i] / iterations);
  }
  fprintf(f, "source node ID, destination node ID, percentage edge was in tree, percentage edge was negative\n");
  for (int v = 0; v < g.nodes; v++) {
    for (int j = g.nindex[v]; j < g.nindex[v + 1]; j++) {
      const int n = g.nlist[j] >> 1;
      if (v < n) {  // only print one copy of each edge (other copy does not have correct negCnt)
        fprintf(f, "%d,%d,%.1f,%.1f\n", g.origID[v], g.origID[n], 100.0 * inTree[j] / iterations, 100.0 * negCnt[j] / iterations);
      }
    }
  }
  fclose(f);

  // finalize
  freeGraph(g);
  delete [] minus;
  delete [] einfo;
  delete [] parent;
  delete [] queue;
  delete [] label;
  delete [] border;
  delete [] inCC;
  delete [] inTree;
  delete [] negCnt;
  delete [] root;

  platform.finalize();
  return 0;
}

