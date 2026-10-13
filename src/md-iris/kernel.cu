#include <cuda_runtime.h>
#include "MD.h"
extern "C" __global__ void md (
  const POSVECTYPE* __restrict__ position,
        FORCEVECTYPE* __restrict__ force,
  const int* __restrict__ neighborList, 
  const int nAtom,
  const int maxNeighbors, 
  const FPTYPE lj1_t,
  const FPTYPE lj2_t,
  const FPTYPE cutsq_t )
{
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nAtom) return;

  POSVECTYPE ipos = position[idx];
  FORCEVECTYPE f = zero;

  int j = 0;
  while (j < maxNeighbors)
  {
    int jidx = neighborList[j*nAtom + idx];

    // Uncoalesced read
    POSVECTYPE jpos = position[jidx];

    // Calculate distance
    FPTYPE delx = ipos.x - jpos.x;
    FPTYPE dely = ipos.y - jpos.y;
    FPTYPE delz = ipos.z - jpos.z;
    FPTYPE r2inv = delx*delx + dely*dely + delz*delz;

    // If distance is less than cutoff, calculate force
    if (r2inv > 0 && r2inv < cutsq_t)
    {
      r2inv = (FPTYPE)1.0 / r2inv;
      FPTYPE r6inv = r2inv * r2inv * r2inv;
      FPTYPE forceC = r2inv * r6inv * (lj1_t * r6inv - lj2_t);

      f.x += delx * forceC;
      f.y += dely * forceC;
      f.z += delz * forceC;
    }
    j++;
  }
  force[idx] = f;
}
