#include <hip/hip_runtime.h>
#include "node.h"

extern "C" __global__ void
Kernel(const Node* __restrict__ d_graph_nodes, 
       const int* __restrict__ d_graph_edges,
       char* __restrict__ d_graph_mask,
       char* __restrict__ d_updatind_graph_mask,
       const char *__restrict__ d_graph_visited,
       int* __restrict__ d_cost,
       const int no_of_nodes) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if( tid<no_of_nodes && d_graph_mask[tid])
  {
    d_graph_mask[tid]=0;
    const int num_edges = d_graph_nodes[tid].no_of_edges;
    const int starting = d_graph_nodes[tid].starting;

    for(int i=starting; i<(num_edges + starting); i++)
    {
      int id = d_graph_edges[i];
      if(!d_graph_visited[id])
      {
        d_cost[id]=d_cost[tid]+1;
        d_updatind_graph_mask[id]=1;
      }
    }
  }
}

extern "C" __global__ void
Kernel2(char* __restrict__ d_graph_mask,
        char* __restrict__ d_updatind_graph_mask,
        char* __restrict__ d_graph_visited,
        char* __restrict__ d_over,
        const int no_of_nodes)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if( tid<no_of_nodes && d_updatind_graph_mask[tid])
  {
    d_graph_mask[tid]=1;
    d_graph_visited[tid]=1;
    *d_over=1;
    d_updatind_graph_mask[tid]=0;
  }
}
