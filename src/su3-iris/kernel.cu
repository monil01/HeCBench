#include <cuda_runtime.h>
#ifndef PRECISION
#define PRECISION 1
#endif
#include "lattice.hpp"
static_assert(sizeof(site)==(PRECISION==1?320:640), "Device site layout");
extern "C" __global__ void k_mat_nn(
  const site*       __restrict__ a,
  const su3_matrix* __restrict__ b,
        site*       __restrict__ c,
  int               total_sites)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int i = id/36;

  if (i < total_sites) {
    int j = (id%36)/9;
    int k = (id%9)/3;
    int l = id%3;
    Complx cc = {0.0, 0.0};
    for (int m=0;m<3;m++)
#ifdef MILC_COMPLEX
      CMULSUM(a[i].link[j].e[k][m], b[j].e[m][l], cc);
#else
      cc += a[i].link[j].e[k][m] * b[j].e[m][l];
#endif
    c[i].link[j].e[k][l] = cc;

  }
}
