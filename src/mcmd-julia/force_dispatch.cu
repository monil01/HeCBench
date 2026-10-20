#include <chrono>
#include <vector>
#include <stdexcept>
#include "force_interface.h"
#define force_kernel mcmd_original_force_kernel
#include "reference_history/force_kernel.cu"
#undef force_kernel
#define CU_OK(call) do{cudaError_t s=(call);if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}while(0)
void force_kernel(const int n,const int block,const int pform,const double cutoff,const double alpha,const int kmax,const int kspace,const double damp,const double*b,const double*rb,d_atom*h){
 static_assert(sizeof(d_atom)==112&&offsetof(d_atom,f)==48&&offsetof(d_atom,molid)==72,"Original atom ABI");
 double *db,*dr;d_atom*da;CU_OK(cudaMalloc(&db,72));CU_OK(cudaMalloc(&dr,72));CU_OK(cudaMalloc(&da,n*sizeof(d_atom)));
 CU_OK(cudaMemcpy(db,b,72,cudaMemcpyHostToDevice));CU_OK(cudaMemcpy(dr,rb,72,cudaMemcpyHostToDevice));CU_OK(cudaMemcpy(da,h,n*sizeof(d_atom),cudaMemcpyHostToDevice));
 auto begin=std::chrono::steady_clock::now();std::vector<d_atom> before(n);double ab[9],ar[9];CU_OK(cudaMemcpy(before.data(),da,n*sizeof(d_atom),cudaMemcpyDeviceToHost));CU_OK(cudaMemcpy(ab,db,72,cudaMemcpyDeviceToHost));CU_OK(cudaMemcpy(ar,dr,72,cudaMemcpyDeviceToHost));mcmd_capture_input(before.data(),n,block,pform,cutoff,alpha,kmax,kspace,damp,ab,ar);mcmd_excluded_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
 int grid=(n+block-1)/block;
 if(mcmd_callback){if(mcmd_callback((double*)da,n,grid,block,pform,cutoff,alpha,kmax,kspace,damp,db,dr))throw std::runtime_error("Authored Julia force callback rejected; native fallback prohibited");}
 else calculateForceKernel<<<grid,block>>>(da,n,cutoff,db,dr,pform,alpha,kmax,kspace,damp);
 CU_OK(cudaGetLastError());CU_OK(cudaMemcpy(h,da,n*sizeof(d_atom),cudaMemcpyDeviceToHost));
 begin=std::chrono::steady_clock::now();mcmd_capture_output(before.data(),h,n,pform,cutoff,alpha,kmax,kspace,ab,ar);mcmd_excluded_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
 CU_OK(cudaFree(da));CU_OK(cudaFree(db));CU_OK(cudaFree(dr));
}
