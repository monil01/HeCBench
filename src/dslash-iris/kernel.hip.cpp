#include "dslash.h"

extern "C" __global__ void make_back(
    const su3_matrix*__restrict__ d_fat,
    const su3_matrix*__restrict__ d_lng,
    const size_t*__restrict__ d_bck, 
    const size_t*__restrict__ d_bck3,
          su3_matrix*__restrict__ d_fatbck,
          su3_matrix*__restrict__ d_lngbck,
    const int total_even_sites)
{
  size_t mySite = blockIdx.x * blockDim.x + threadIdx.x;
  if (mySite < total_even_sites) {
    for(int dir = 0; dir < 4; dir++) {
      su3_adjoint( d_fat + 4*d_bck[4*mySite+dir]+dir, 
          d_fatbck + 4*mySite+dir );
      su3_adjoint( d_lng + 4*d_bck3[4*mySite+dir]+dir, 
          d_lngbck + 4*mySite+dir );
    }
  }
}

extern "C" __global__ void dslash (
    const su3_matrix*__restrict__ d_fat,
    const su3_matrix*__restrict__ d_lng,
    const su3_matrix*__restrict__ d_fatbck,
    const su3_matrix*__restrict__ d_lngbck,
    const su3_vector*__restrict__ d_src,
          su3_vector*__restrict__ d_dst,
    const size_t*__restrict__ d_fwd,
    const size_t*__restrict__ d_bck,
    const size_t*__restrict__ d_fwd3,
    const size_t*__restrict__ d_bck3,
    const int total_even_sites)
{
  size_t mySite = blockIdx.x * blockDim.x + threadIdx.x;
  if (mySite < total_even_sites) {
    su3_vector v;
    for (size_t k=0; k<4; ++k) {
      auto a = d_fat + mySite*4 + k;
      auto b = d_src + d_fwd[4*mySite + k];
      if (k == 0)
        mult_su3_mat_vec(a, b, &d_dst[mySite]);
      else 
        mult_su3_mat_vec_sum(a, b, &d_dst[mySite]);
    }
    for (size_t k=0; k<4; ++k) {
      auto a = d_lng + mySite*4 + k;
      auto b = d_src + d_fwd3[4*mySite + k];
      if (k == 0) 
        mult_su3_mat_vec(a, b, &v);
      else
        mult_su3_mat_vec_sum(a, b, &v);
    }
    add_su3_vector(&d_dst[mySite], &v, &d_dst[mySite]);
    for (size_t k=0; k<4; ++k) {
      auto a = d_fatbck + mySite*4 + k;
      auto b = d_src + d_bck[4*mySite + k];
      if (k == 0) 
        mult_su3_mat_vec(a, b, &v);
      else
        mult_su3_mat_vec_sum(a, b, &v);
    }
    sub_su3_vector(&d_dst[mySite], &v, &d_dst[mySite]);
    for (size_t k=0; k<4; ++k) {
      auto a = d_lngbck + mySite*4 + k;
      auto b = d_src + d_bck3[4*mySite + k];
      if (k == 0) 
        mult_su3_mat_vec(a, b, &v);
      else
        mult_su3_mat_vec_sum(a, b, &v);
    }
    sub_su3_vector(&d_dst[mySite], &v, &d_dst[mySite]);
  } // end of if mySite
}
