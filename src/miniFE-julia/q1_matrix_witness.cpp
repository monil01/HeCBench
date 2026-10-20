#define main minife_original_main
#include "reference_history/src/main.cpp"
#undef main
#include "q1_oracle.h"
#include <cstdio>
int main(){bool all=true;printf("{\"cases\":[");int ci=0;
 for(int dim:{2,4,16}){
  Box box={0,dim,0,dim,0,dim};miniFE::simple_mesh_description<int>mesh(box,box);miniFE::CSRMatrix<double,int,int>A;miniFE::generate_matrix_structure(mesh,A);miniFE::Vector<double,int,int>b(0,A.rows.size());miniFE::Parameters par;miniFE::assemble_FE_data(mesh,A,b,par);auto rawA=A;auto rawb=b;
  for(auto hot:mesh.bc_rows_1)mesh.bc_rows_0.erase(hot);miniFE::impose_dirichlet(0.,A,b,dim+1,dim+1,dim+1,mesh.bc_rows_0);miniFE::impose_dirichlet(1.,A,b,dim+1,dim+1,dim+1,mesh.bc_rows_1);
  Q1Input v;v.dim=dim;v.n=A.rows.size();v.nz=A.packed_cols.size();v.rp=A.row_offsets;v.ci=A.packed_cols;v.A=A.packed_coefs;v.b=b.coefs;auto review=q1_review_matrix(v);
  long double qerr=0,lerr=0;int s=dim+1;
  for(int row=0;row<v.n;row++){
   int x=row%s,y=(row/s)%s,z=row/(s*s);if(q1_boundary(x,y,z,dim))continue;long double aq=0,al=0;
   for(int j=rawA.row_offsets[row];j<rawA.row_offsets[row+1];j++){
    int c=rawA.packed_cols[j];long double xx=(long double)(c%s)/dim,yy=(long double)((c/s)%s)/dim,zz=(long double)(c/(s*s))/dim;
    aq+=(long double)rawA.packed_coefs[j]*(xx*(1-xx)/2);al+=(long double)rawA.packed_coefs[j]*(xx+2*yy+3*zz);
   }qerr=std::max(qerr,fabsl(aq-rawb.coefs[row]));lerr=std::max(lerr,fabsl(al));
  }
  bool pass=review.pass&&qerr<2e-13L&&lerr<2e-13L;all&=pass;
  printf("%s{\"cells_per_dimension\":%d,\"nodes\":%d,\"full_CSR_entries_checked\":%lld,\"max_coefficient_error_vs_independent_tensor_operator\":%.17g,\"max_corrected_RHS_error\":%.17g,\"unit_Poisson_quadratic_patch_max_abs_error\":%.17g,\"harmonic_linear_patch_max_abs_error\":%.17g,\"pass\":%s}",ci++?",":"",dim,v.n,review.entries,review.max_A_abs,review.max_b_abs,(double)qerr,(double)lerr,pass?"true":"false");
 }printf("],\"all_pass\":%s,\"quadrature_coordinate\":0.577350269,\"scope\":\"Independent tensor operator and manufactured polynomial identities; corrected hot-face priority; source-rounded quadrature, no GPU calls in witness.\"}\n",all?"true":"false");return all?0:1;
}
