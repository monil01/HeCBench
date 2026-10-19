// Minimal independently justified corrected-native secondary, graphB+.
// Original two-phase initMinus has cross-block writes to the same minus[j].
// Each edge has one owner here; source intended tree sign/default true unchanged.
static __global__ void initMinus(
 const int edges,const int nodes,const int*const __restrict__ nindex,
 const int*const __restrict__ nlist,const EdgeInfo*const einfo,bool*const minus)
{
 const int from=threadIdx.x+blockIdx.x*ThreadsPerBlock;
 const int incr=gridDim.x*ThreadsPerBlock;
 for(int j=from;j<edges;j+=incr)
  minus[j]=(nlist[j]&1)?bool(einfo[j].end&1):true;
}
