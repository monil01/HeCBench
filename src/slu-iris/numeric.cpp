#include <iris/iris.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <initializer_list>
#include "symbolic.h"
#include "Timer.h"
using namespace std;
#define TMPMEMNUM 10353
#define Nstreams 16
struct LaunchDim { size_t x,y; LaunchDim(size_t a,size_t b=1):x(a),y(b){} };
struct Argument { void *pointer; int info; };
static Argument M(iris::DMem& mem,int mode) { return {&mem,mode}; }
template<class T> static Argument S(const T& value) { return {const_cast<T*>(&value),sizeof(T)}; }
static void require_iris(int code) { if(code!=IRIS_SUCCESS) throw runtime_error("IRIS LU task failure"); }
static const char *variant(const char *base,size_t bytes) {
  if(string(base)=="RL") {
    if(bytes==32*sizeof(REAL)) return "RL_s32";
    if(bytes==4*sizeof(REAL)) return "RL_s4";
    if(bytes==2*sizeof(REAL)) return "RL_s2";
  } else {
    if(bytes==32*sizeof(REAL)) return "RL_perturb_s32";
    if(bytes==4*sizeof(REAL)) return "RL_perturb_s4";
    if(bytes==2*sizeof(REAL)) return "RL_perturb_s2";
  }
  throw runtime_error("Unexpected original RL shared memory size");
}
static void launch(const char *name,size_t blocks,size_t threads,initializer_list<Argument> args) {
  vector<void*> pointers; vector<int> info;
  for(const auto& arg:args) { pointers.push_back(arg.pointer); info.push_back(arg.info); }
  size_t global[]={blocks*threads},local[]={threads};
  iris::Task task;
  require_iris(task.kernel(name,1,nullptr,global,local,args.size(),pointers.data(),info.data()));
  require_iris(task.submit(iris_gpu,nullptr,1));
}
void LUonDevice(Symbolic_Matrix &A_sym, ostream &out, ostream &err, bool PERTURB)
{
  unsigned n = A_sym.n;
  unsigned nnz = A_sym.nnz;
  unsigned num_lev = A_sym.num_lev;
  iris::DMem sym_c_ptr_dev(A_sym.sym_c_ptr.data(), (n+1)*sizeof(unsigned));
  iris::DMem sym_r_idx_dev(A_sym.sym_r_idx.data(), nnz*sizeof(unsigned));
  iris::DMem val_dev(A_sym.val.data(), nnz*sizeof(REAL));
  iris::DMem l_col_ptr_dev(A_sym.l_col_ptr.data(), n*sizeof(unsigned));
  iris::DMem csr_r_ptr_dev(A_sym.csr_r_ptr.data(), (n+1)*sizeof(unsigned));
  iris::DMem csr_c_idx_dev(A_sym.csr_c_idx.data(), nnz*sizeof(unsigned));
  iris::DMem csr_diag_ptr_dev(A_sym.csr_diag_ptr.data(), n*sizeof(unsigned));
  iris::DMem level_idx_dev(A_sym.level_idx.data(), n*sizeof(int));
  std::vector<REAL> scratch(static_cast<size_t>(TMPMEMNUM)*n, REAL(0));
  iris::DMem tmpMem(scratch.data(), scratch.size()*sizeof(REAL));

  // calculate 1-norm of A and perturbation value for perturbation
  float pert = 0;
  if (PERTURB)
  {
    float norm_A = 0;
    for (unsigned i = 0; i < n; ++i)
    {
      float tmp = 0;
      for (unsigned j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j)
        tmp += abs(A_sym.val[j]);
      if (norm_A < tmp)
        norm_A = tmp;
    }
    pert = 3.45e-4 * norm_A;
    out << "Gaussian elimination with static pivoting (GESP)..." << endl;
    out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << endl;
  }



  Timer t;
  double utime;
  t.start();
  for (unsigned i = 0; i < num_lev; ++i)
  {
    int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];

    if (lev_size > 896) { //3584 / 4
      unsigned WarpsPerBlock = 2;
      LaunchDim dimBlock(WarpsPerBlock * 32, 1);
      size_t MemSize = WarpsPerBlock * sizeof(REAL);

      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        LaunchDim dimGrid(restCol, 1);
        if (!PERTURB)
          launch(variant("RL", MemSize), dimGrid.x, dimBlock.x, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(val_dev,iris_rw), M(l_col_ptr_dev,iris_r), M(csr_r_ptr_dev,iris_r), M(csr_c_idx_dev,iris_r), M(csr_diag_ptr_dev,iris_r), M(level_idx_dev,iris_r), M(tmpMem,iris_rw), S(n), S(A_sym.level_ptr[i]), S(j*TMPMEMNUM)});
        else
          launch(variant("RL_perturb", MemSize), dimGrid.x, dimBlock.x, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(val_dev,iris_rw), M(l_col_ptr_dev,iris_r), M(csr_r_ptr_dev,iris_r), M(csr_c_idx_dev,iris_r), M(csr_diag_ptr_dev,iris_r), M(level_idx_dev,iris_r), M(tmpMem,iris_rw), S(n), S(A_sym.level_ptr[i]), S(j*TMPMEMNUM), S(pert)});
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else if (lev_size > 448) {
      unsigned WarpsPerBlock = 4;
      LaunchDim dimBlock(WarpsPerBlock * 32, 1);
      size_t MemSize = WarpsPerBlock * sizeof(REAL);

      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        LaunchDim dimGrid(restCol, 1);
        if (!PERTURB)
          launch(variant("RL", MemSize), dimGrid.x, dimBlock.x, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(val_dev,iris_rw), M(l_col_ptr_dev,iris_r), M(csr_r_ptr_dev,iris_r), M(csr_c_idx_dev,iris_r), M(csr_diag_ptr_dev,iris_r), M(level_idx_dev,iris_r), M(tmpMem,iris_rw), S(n), S(A_sym.level_ptr[i]), S(j*TMPMEMNUM)});
        else
          launch(variant("RL_perturb", MemSize), dimGrid.x, dimBlock.x, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(val_dev,iris_rw), M(l_col_ptr_dev,iris_r), M(csr_r_ptr_dev,iris_r), M(csr_c_idx_dev,iris_r), M(csr_diag_ptr_dev,iris_r), M(level_idx_dev,iris_r), M(tmpMem,iris_rw), S(n), S(A_sym.level_ptr[i]), S(j*TMPMEMNUM), S(pert)});
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else if (lev_size > Nstreams) {
      LaunchDim dimBlock(256, 1);
      size_t MemSize = 32 * sizeof(REAL);
      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        LaunchDim dimGrid(restCol, 1);
        if (!PERTURB)
          launch(variant("RL", MemSize), dimGrid.x, dimBlock.x, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(val_dev,iris_rw), M(l_col_ptr_dev,iris_r), M(csr_r_ptr_dev,iris_r), M(csr_c_idx_dev,iris_r), M(csr_diag_ptr_dev,iris_r), M(level_idx_dev,iris_r), M(tmpMem,iris_rw), S(n), S(A_sym.level_ptr[i]), S(j*TMPMEMNUM)});
        else
          launch(variant("RL_perturb", MemSize), dimGrid.x, dimBlock.x, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(val_dev,iris_rw), M(l_col_ptr_dev,iris_r), M(csr_r_ptr_dev,iris_r), M(csr_c_idx_dev,iris_r), M(csr_diag_ptr_dev,iris_r), M(level_idx_dev,iris_r), M(tmpMem,iris_rw), S(n), S(A_sym.level_ptr[i]), S(j*TMPMEMNUM), S(pert)});
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else { // "Big" levels
      for (int offset = 0; offset < lev_size; offset += Nstreams) {
        for (int j = 0; j < Nstreams; j++) {
          if (j + offset < lev_size) {
            const unsigned currentCol = A_sym.level_idx[A_sym.level_ptr[i] + j + offset];
            const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
              - A_sym.csr_diag_ptr[currentCol] - 1;

            if (!PERTURB)
              launch("RL_onecol_factorizeCurrentCol", 1, 256, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(val_dev,iris_rw), M(l_col_ptr_dev,iris_r), S(currentCol), M(tmpMem,iris_rw), S(j), S(n)});
            else
              launch("RL_onecol_factorizeCurrentCol_perturb", 1, 256, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(val_dev,iris_rw), M(l_col_ptr_dev,iris_r), S(currentCol), M(tmpMem,iris_rw), S(j), S(n), S(pert)});
            if (subMatSize > 0)
              launch("RL_onecol_updateSubmat", subMatSize, 256, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(val_dev,iris_rw), M(csr_c_idx_dev,iris_r), M(csr_diag_ptr_dev,iris_r), S(currentCol), M(tmpMem,iris_rw), S(j), S(n)});
            launch("RL_onecol_cleartmpMem", 1, 256, {M(sym_c_ptr_dev,iris_r), M(sym_r_idx_dev,iris_r), M(l_col_ptr_dev,iris_r), S(currentCol), M(tmpMem,iris_rw), S(j), S(n)});
          }
        }
      }
    }
  }

  t.elapsedUserTime(utime);
  out << "Total LU kernel execution time: " << utime << " ms" << std::endl;

  //copy LU val back to main mem
  iris::Task flush;
  require_iris(flush.flush_out(val_dev));
  require_iris(flush.submit(iris_gpu,nullptr,1));

#ifdef VERIFY
  //check NaN elements
  unsigned err_find = 0;
  for(unsigned i = 0; i < nnz; i++)
    if(isnan(A_sym.val[i]) || isinf(A_sym.val[i]))
      err_find++;

  if (err_find != 0)
    err << "LU data check: NaN found!!" << std::endl;
#endif

}
