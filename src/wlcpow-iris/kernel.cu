// Modified for IRIS DMem/HIP on 2026-10-13; GNU GPL v3, see LICENSE.
#include <cuda_runtime.h>
#include "utils.h"
extern "C" __global__
void bond_wlcpowallvisc(
             r64* __restrict__ force_x,
             r64* __restrict__ force_y,
             r64* __restrict__ force_z,
    const float4* __restrict__ coord_merged,
    const float4* __restrict__ veloc,
    const int*  __restrict__ nbond,
    const int2* __restrict__ bonds,
    const r64* __restrict__ bond_r0,
    const r32* __restrict__ temp_global,
    const r32* __restrict__ r0_global,
    const r32* __restrict__ mu_targ_global,
    const r32* __restrict__ qp_global,
    const r32* __restrict__ gamc_global,
    const r32* __restrict__ gamt_global,
    const r32* __restrict__ sigc_global,
    const r32* __restrict__ sigt_global,
    const float3 period,
    const int padding,
    const int n_type,
    const int n_local )
{
  __shared__ r32 shared_data[33 * 8];
  r32* temp    = &shared_data[0];
  r32* r0      = &shared_data[1*(n_type+1)];
  r32* mu_targ = &shared_data[2*(n_type+1)];
  r32* qp      = &shared_data[3*(n_type+1)];
  r32* gamc    = &shared_data[4*(n_type+1)];
  r32* gamt    = &shared_data[5*(n_type+1)];
  r32* sigc    = &shared_data[6*(n_type+1)];
  r32* sigt    = &shared_data[7*(n_type+1)];

  for ( int i = threadIdx.x; i < n_type + 1; i += blockDim.x ) {
    temp[i]    = temp_global[i];
    r0[i]      = r0_global[i];
    mu_targ[i] = mu_targ_global[i];
    qp[i]      = qp_global[i];
    gamc[i]    = gamc_global[i];
    gamt[i]    = gamt_global[i];
    sigc[i]    = sigc_global[i];
    sigt[i]    = sigt_global[i];
  }
  __syncthreads();

  for( int i = blockIdx.x * blockDim.x + threadIdx.x;
           i < n_local ; i += gridDim.x * blockDim.x ) {
    int n = nbond[i];
    float4 coord1 = coord_merged[i];
    float4 veloc1 = veloc[i];
    r32 fxi = 0.f, fyi = 0.f, fzi = 0.f;

    for( int p = 0; p < n; p++ ) {
      int j = bonds[ i + p*padding ].x;
      int type = bonds[ i + p*padding ].y;
      float4 coord2 = coord_merged[j];
      r32 delx = minimum_image( coord1.x - coord2.x, period.x );
      r32 dely = minimum_image( coord1.y - coord2.y, period.y );
      r32 delz = minimum_image( coord1.z - coord2.z, period.z );
      float4 veloc2 = veloc[j];
      r32 dvx = veloc1.x - veloc2.x;
      r32 dvy = veloc1.y - veloc2.y;
      r32 dvz = veloc1.z - veloc2.z;

      r32 l0 = bond_r0[ i + p*padding ];
      r32 ra = sqrtf(delx*delx + dely*dely + delz*delz);
      r32 lmax = l0*r0[type];
      r32 rr = 1.0f/r0[type];
      r32 sr = (1.0f-rr)*(1.0f-rr);
      r32 kph = powf(l0,qp[type])*temp[type]*(0.25f/sr-0.25f+rr);
      // mu is described in the papers
      r32 mu = 0.433f*(   // 0.25 * sqrt(3)
	       temp[type]*(-0.25f/sr + 0.25f + 
               0.5f*rr/(sr*(1.0f-rr)))/(lmax*rr) +
               kph*(qp[type]+1.0f)/powf(l0,qp[type]+1.0f));
      r32 lambda = mu/mu_targ[type];
      kph = kph/lambda;
      rr = ra/lmax;
      r32 rlogarg = powf(ra,qp[type]+1.0f);
      r32 vv = (delx*dvx + dely*dvy + delz*dvz)/ra;

      if (rr >= 0.99) rr = 0.99f;
      if (rlogarg < 0.01) rlogarg = 0.01f;

      float4 wrr;
      r32 ww[3][3];

      for (int tes=0; tes<3; tes++) {
        for (int see=0; see<3; see++) {
          int v1 = __float_as_int(veloc1.w);
          int v2 = __float_as_int(veloc2.w);
          ww[tes][see] = gaussian_TEA_fast<4>(v1 > v2, v1+tes, v2+see);
        }
      }

      wrr.w = (ww[0][0]+ww[1][1]+ww[2][2])/3.0f;
      wrr.x = (ww[0][0]-wrr.w)*delx + 0.5f*(ww[0][1]+ww[1][0])*dely + 0.5f*(ww[0][2]+ww[2][0])*delz;
      wrr.y = 0.5f*(ww[1][0]+ww[0][1])*delx + (ww[1][1]-wrr.w)*dely + 0.5f*(ww[1][2]+ww[2][1])*delz;
      wrr.z = 0.5f*(ww[2][0]+ww[0][2])*delx + 0.5f*(ww[2][1]+ww[1][2])*dely + (ww[2][2]-wrr.w)*delz;

      r32 fforce = - temp[type]*(0.25f/(1.0f-rr)/(1.0f-rr)-0.25f+rr)/lambda/ra + kph/rlogarg + (sigc[type]*wrr.w - gamc[type]*vv)/ra;
      r32 fxij = delx*fforce - gamt[type]*dvx + sigt[type]*wrr.x/ra;
      r32 fyij = dely*fforce - gamt[type]*dvy + sigt[type]*wrr.y/ra;
      r32 fzij = delz*fforce - gamt[type]*dvz + sigt[type]*wrr.z/ra;

      fxi += fxij;
      fyi += fyij;
      fzi += fzij;
    }
    force_x[i] += fxi;
    force_y[i] += fyi;
    force_z[i] += fzi;
  }
}
