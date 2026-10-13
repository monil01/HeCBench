#include <cuda_runtime.h>
#include "point_struct.h"
extern "C" __global__ void compute_cost(const Point_Struct *p, const float *coord,
    float *work, const int *table, char *membership, int num, int dim, long x, int K) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= num) return;
  // Direct reads retain the source's per-dimension arithmetic order.
  float cost=0;
  for(int d=0; d<dim; ++d) {
    float delta=coord[d*num+i]-coord[d*num+x];
    cost += delta*delta;
  }
  cost *= p[i].weight;
  int base=i*(K+1);
  if(cost<p[i].cost) { membership[i]='1'; work[base+K]=cost-p[i].cost; }
  else work[base+table[p[i].assign]] += p[i].cost-cost;
}
