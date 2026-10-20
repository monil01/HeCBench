#define main minife_original_main
#include "reference_history/src/main.cpp"
#undef main
#include <iomanip>
int main(){
 Box box={0,2,0,2,0,2};miniFE::simple_mesh_description<int> mesh(box,box);miniFE::CSRMatrix<double,int,int>A;miniFE::generate_matrix_structure(mesh,A);miniFE::Vector<double,int,int>b(0,A.rows.size());miniFE::Parameters params;miniFE::assemble_FE_data(mesh,A,b,params);
 auto rawA=A;auto rawb=b;auto correctedA=A;auto correctedb=b;
 miniFE::impose_dirichlet(0.,A,b,3,3,3,mesh.bc_rows_0);miniFE::impose_dirichlet(1.,A,b,3,3,3,mesh.bc_rows_1);
 auto zero=mesh.bc_rows_0;size_t overlap=0;for(int row:mesh.bc_rows_1)overlap+=zero.erase(row);
 miniFE::impose_dirichlet(0.,correctedA,correctedb,3,3,3,zero);miniFE::impose_dirichlet(1.,correctedA,correctedb,3,3,3,mesh.bc_rows_1);
 double origerr=0,correrr=0;int count=0;
 for(int i=0;i<(int)A.rows.size();i++){
  bool hot=mesh.bc_rows_1.count(i),cold=mesh.bc_rows_0.count(i);double expected=hot?1.:cold?0.:rawb.coefs[i];
  if(!hot&&!cold){for(int j=rawA.row_offsets[i];j<rawA.row_offsets[i+1];j++)if(mesh.bc_rows_1.count(rawA.packed_cols[j]))expected-=rawA.packed_coefs[j];count++;}
  origerr=std::max(origerr,std::abs(b.coefs[i]-expected));correrr=std::max(correrr,std::abs(correctedb.coefs[i]-expected));
 }
 double diag=0;for(int j=A.row_offsets[13];j<A.row_offsets[14];j++)if(A.packed_cols[j]==13)diag=A.packed_coefs[j];
 printf("{\"source_elements\":2,\"nodes\":27,\"overlapping_hot_cold_boundary_nodes\":%zu,\"independent_interior_rows_checked\":%d,\"raw_source_Q1_interior_load\":%.17g,\"original_center_rhs\":%.17g,\"priority_consistent_center_rhs\":%.17g,\"center_diagonal\":%.17g,\"original_center_discrete_solution\":%.17g,\"corrected_center_discrete_solution\":%.17g,\"original_max_rhs_error\":%.17g,\"corrected_max_rhs_error\":%.17g,\"conclusion\":\"OriginalzeroBCcolumneliminationloseshotfaceedgecontributionsbeforehotBCoverride; excludinghotrowsfromzeroBCsetrestoresdirectAfree,bc*gidentity. SeparateQ1forcingloadQ1isinoriginalsource; homogeneousanalyticcheckeromitsit.\"}\n",overlap,count,rawb.coefs[13],b.coefs[13],correctedb.coefs[13],diag,b.coefs[13]/diag,correctedb.coefs[13]/diag,origerr,correrr);
 return origerr>.1&&correrr<1e-14&&overlap==8?0:1;
}
