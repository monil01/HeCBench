#pragma once
#include <cstdlib>
#include <fstream>
#include <string>
#include <stdexcept>
#include "symbolic.h"
template<class T> static void slu_array(const std::string& path,const std::vector<T>& data) {
  std::ofstream out(path,std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
  if(!out) throw std::runtime_error("Failed writing SLU validation artifact "+path);
}
static void slu_export(const Symbolic_Matrix& a,const std::vector<REAL>& x) {
  const char *dir=std::getenv("SLU_VERIFY_DIR");
  if(!dir) return;
  const std::string prefix=std::string(dir)+"/";
  std::ofstream metadata(prefix+"metadata.json");
  metadata<<"{\"n\":"<<a.n<<",\"nnz\":"<<a.nnz<<",\"levels\":"<<a.num_lev<<"}\n";
  if(!metadata) throw std::runtime_error("Failed writing SLU validation metadata");
  slu_array(prefix+"lu.f32",a.val);
  slu_array(prefix+"solution.f32",x);
  slu_array(prefix+"sym_c_ptr.u32",a.sym_c_ptr);
  slu_array(prefix+"sym_r_idx.u32",a.sym_r_idx);
  slu_array(prefix+"l_col_ptr.u32",a.l_col_ptr);
  slu_array(prefix+"csr_r_ptr.u32",a.csr_r_ptr);
  slu_array(prefix+"csr_c_idx.u32",a.csr_c_idx);
  slu_array(prefix+"csr_diag_ptr.u32",a.csr_diag_ptr);
  slu_array(prefix+"level_idx.i32",a.level_idx);
  slu_array(prefix+"level_ptr.i32",a.level_ptr);
}
