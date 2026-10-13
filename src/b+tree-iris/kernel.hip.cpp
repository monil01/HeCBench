#include <hip/hip_runtime.h>
#include "./common.h"

extern "C" __global__ void findK(const long height,
                                  const knode *__restrict__ knodesD,
                                  const long knodes_elem,
                                  const record *__restrict__ recordsD,
                                  long *__restrict__ currKnodeD,
                                  long *__restrict__ offsetD,
                                  const int *__restrict__ keysD,
                                  record *ansD) {
  int thid = threadIdx.x;
  int bid = blockIdx.x;
  for (int i = 0; i < height; i++) {
    if (knodesD[currKnodeD[bid]].keys[thid] <= keysD[bid] &&
        knodesD[currKnodeD[bid]].keys[thid + 1] > keysD[bid]) {
      if (knodesD[offsetD[bid]].indices[thid] < knodes_elem)
        offsetD[bid] = knodesD[offsetD[bid]].indices[thid];
    }
    __syncthreads();
    if (thid == 0) currKnodeD[bid] = offsetD[bid];
    __syncthreads();
  }
  if (knodesD[currKnodeD[bid]].keys[thid] == keysD[bid])
    ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
}

extern "C" __global__ void findRangeK(const long height,
                                       const knode *__restrict__ knodesD,
                                       const long knodes_elem,
                                       long *__restrict__ currKnodeD,
                                       long *__restrict__ offsetD,
                                       long *__restrict__ lastKnodeD,
                                       long *__restrict__ offset_2D,
                                       const int *__restrict__ startD,
                                       const int *__restrict__ endD,
                                       int *__restrict__ RecstartD,
                                       int *__restrict__ ReclenD) {
  int thid = threadIdx.x;
  int bid = blockIdx.x;
  for (int i = 0; i < height; i++) {
    if (knodesD[currKnodeD[bid]].keys[thid] <= startD[bid] &&
        knodesD[currKnodeD[bid]].keys[thid + 1] > startD[bid]) {
      if (knodesD[currKnodeD[bid]].indices[thid] < knodes_elem)
        offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
    }
    if (knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid] &&
        knodesD[lastKnodeD[bid]].keys[thid + 1] > endD[bid]) {
      if (knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem)
        offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
    }
    __syncthreads();
    if (thid == 0) {
      currKnodeD[bid] = offsetD[bid];
      lastKnodeD[bid] = offset_2D[bid];
    }
    __syncthreads();
  }
  if (knodesD[currKnodeD[bid]].keys[thid] == startD[bid])
    RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
  __syncthreads();
  if (knodesD[lastKnodeD[bid]].keys[thid] == endD[bid])
    ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid] + 1;
}
