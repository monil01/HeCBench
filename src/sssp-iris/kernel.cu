/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba,
 *        University of Illinois nor the names of its contributors may be used
 *        to endorse or promote products derived from this Software without
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <cuda_runtime.h>
#include "support/common.h"
extern "C" __global__ void SSSP_gpu(
    const Node *__restrict__ graph_nodes_av,
    const Edge *__restrict__ graph_edges_av,
    int *__restrict__ cost,
    int *__restrict__ color,
    const int *__restrict__ q1,
          int *__restrict__ q2,
    const int *__restrict__ n_t,
    int *__restrict__ head,
    int *__restrict__ tail,
    int *__restrict__ overflow,
    const int *__restrict__ gray_shade,
    int *__restrict__ iter)
{
  __shared__ int l_mem[W_QUEUE_SIZE+2];
  __shared__ int tail_bin;
  int* l_q2 = l_mem;
  int* shift = l_mem + W_QUEUE_SIZE;
  int* base = l_mem + W_QUEUE_SIZE + 1;

  const int tid     = threadIdx.x;
  const int gtid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int WG_SIZE = blockDim.x;

  int n_t_local = *n_t; // atomicAdd(n_t, 0);
  int gray_shade_local = *gray_shade; // atomicAdd(&gray_shade[0], 0);

  if(tid == 0) {
    // Reset queue
    tail_bin = 0;
  }

  // Fetch frontier elements from the queue
  if(tid == 0)
    *base = atomicAdd(&head[0], WG_SIZE);
  __syncthreads();

  int my_base = *base;
  while(my_base < n_t_local) {

    // If local queue might overflow
    if(tail_bin >= W_QUEUE_SIZE / 2) {
      if(tid == 0) {
        // Add local tail_bin to tail
        *shift = atomicAdd(&tail[0], tail_bin);
      }
      __syncthreads();
      int local_shift = tid;
      while(local_shift < tail_bin) {
        q2[*shift + local_shift] = l_q2[local_shift];
        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
        local_shift += WG_SIZE;
      }
      __syncthreads();
      if(tid == 0) {
        // Reset local queue
        tail_bin = 0;
      }
      __syncthreads();
    }

    __syncthreads(); // l_q2 read complete

    if(my_base + tid < n_t_local && *overflow == 0) {
      // Visit a node from the current frontier
      int pid = q1[my_base + tid];
      //////////////// Visit node ///////////////////////////
      atomicExch(&color[pid], BLACK); // Node visited
      int  cur_cost = cost[pid]; // atomicAdd(&cost[pid], 0); // Look up shortest-path distance to this node
      Node cur_node;
      cur_node.x = graph_nodes_av[pid].x;
      cur_node.y = graph_nodes_av[pid].y;
      Edge cur_edge;
      // For each outgoing edge
      for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
        cur_edge.x = graph_edges_av[i].x;
        cur_edge.y = graph_edges_av[i].y;
        int id     = cur_edge.x;
        int cost_local   = cur_edge.y;
        cost_local += cur_cost;
        int orig_cost = atomicMax(&cost[id], cost_local);
        if(orig_cost < cost_local) {
          int old_color = atomicMax(&color[id], gray_shade_local);
          if(old_color != gray_shade_local) {
            // Push to the queue
            int tail_index = atomicAdd(&tail_bin, 1);
            if(tail_index >= W_QUEUE_SIZE) {
              *overflow = 1;
            } else
              l_q2[tail_index] = id;
          }
        }
      }
    }

    if(tid == 0)
      *base = atomicAdd(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
    __syncthreads();
    my_base = *base;
  }
  /////////////////////////////////////////////////////////
  // Compute size of the output and allocate space in the global queue
  if(tid == 0) {
    *shift = atomicAdd(&tail[0], tail_bin);
  }
  __syncthreads();
  ///////////////////// CONCATENATE INTO GLOBAL MEMORY /////////////////////
  int local_shift = tid;
  while(local_shift < tail_bin) {
    q2[*shift + local_shift] = l_q2[local_shift];
    // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
    local_shift += WG_SIZE;
  }
  //////////////////////////////////////////////////////////////////////////

  if(gtid == 0) {
    atomicAdd(&iter[0], 1);
  }
}


