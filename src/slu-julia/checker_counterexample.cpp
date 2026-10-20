// CPU-only acceptance-gate counterexample, NOT an observed GPU failure.
#include <iostream>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include "symbolic.h"
#include "independent_oracle.h"
int main(){std::ostringstream out,err;Symbolic_Matrix m(2,out,err);m.nnz=3;m.sym_c_ptr={0,2,3};m.sym_r_idx={0,1,1};m.l_col_ptr={0,2};m.val={1,0,1};m.ABFTCalculateCCA();m.val={.5f,1,1};m.ABFTCheckResult();bool abft=out.str().find("Results passed ABFT check.")!=std::string::npos;std::string path="/tmp/hecbench-stencil-ports-slu-ABFT-identity2-"+std::to_string(getpid())+".mtx";std::ofstream f(path);f<<"%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n2 2 1\n";f.close();auto r=independent_lu(m.sym_c_ptr,m.sym_r_idx,m.l_col_ptr,std::vector<float>{1,0,1},m.val,std::vector<float>{1,1},path);std::cout<<"{\"CPU_counterfactual_only\":true,\"observed_wrong_GPU_result\":false,\"original_ABFT_checker_pass\":"<<(abft?"true":"false")<<",\"independent_full_product_pass\":"<<(r.pass?"true":"false")<<",\"maximum_factor_componentwise_backward_error\":"<<r.componentwise<<",\"factor_normwise_backward_error\":"<<r.normwise<<",\"matrix\":\""<<path<<"\",\"original_checker_message\":\"Results passed ABFT check.\"}\n";return abft&&!r.pass?0:1;}
