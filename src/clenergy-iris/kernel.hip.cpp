#include <hip/hip_runtime.h>
#define MAXATOMS 4000

#define UNROLLX       8
#define UNROLLY       1
#define BLOCKSIZEX    8
#define BLOCKSIZEY    8
#define BLOCKSIZE    BLOCKSIZEX * BLOCKSIZEY

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
extern "C" __global__ void cenergy(const int numatoms, const float gridspacing,
                        float *energygrid, const float4 *atominfo)
{
  unsigned int xindex  = blockIdx.x * blockDim.x * UNROLLX + threadIdx.x;
  unsigned int yindex  = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int outaddr = gridDim.x * blockDim.x * UNROLLX * yindex + xindex;

  float coory = gridspacing * yindex;
  float coorx = gridspacing * xindex;

  float energyvalx1=0.0f;
  float energyvalx2=0.0f;
  float energyvalx3=0.0f;
  float energyvalx4=0.0f;
  float energyvalx5=0.0f;
  float energyvalx6=0.0f;
  float energyvalx7=0.0f;
  float energyvalx8=0.0f;

  float gridspacing_u = gridspacing * BLOCKSIZEX;

  //
  // XXX 59/8 FLOPS per atom
  //
  int atomid;
  for (atomid=0; atomid<numatoms; atomid++) {
    float dy = coory - atominfo[atomid].y;
    float dyz2 = (dy * dy) + atominfo[atomid].z;

    float dx1 = coorx - atominfo[atomid].x;
    float dx2 = dx1 + gridspacing_u;
    float dx3 = dx2 + gridspacing_u;
    float dx4 = dx3 + gridspacing_u;
    float dx5 = dx4 + gridspacing_u;
    float dx6 = dx5 + gridspacing_u;
    float dx7 = dx6 + gridspacing_u;
    float dx8 = dx7 + gridspacing_u;

    energyvalx1 += atominfo[atomid].w * rsqrtf(dx1*dx1 + dyz2);
    energyvalx2 += atominfo[atomid].w * rsqrtf(dx2*dx2 + dyz2);
    energyvalx3 += atominfo[atomid].w * rsqrtf(dx3*dx3 + dyz2);
    energyvalx4 += atominfo[atomid].w * rsqrtf(dx4*dx4 + dyz2);
    energyvalx5 += atominfo[atomid].w * rsqrtf(dx5*dx5 + dyz2);
    energyvalx6 += atominfo[atomid].w * rsqrtf(dx6*dx6 + dyz2);
    energyvalx7 += atominfo[atomid].w * rsqrtf(dx7*dx7 + dyz2);
    energyvalx8 += atominfo[atomid].w * rsqrtf(dx8*dx8 + dyz2);
  }

  energygrid[outaddr             ] += energyvalx1;
  energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
  energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
  energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
  energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
  energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
  energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
  energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
}
