#define main original_amgmk_main
#include "../../amgmk-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
static void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
int main(){int nx=50,ny=50,nz=50,N=nx*ny*nz;double v[4]={6,-1,-1,-1};hypre_Vector *rhs,*x,*sol;hypre_CSRMatrix*A=GenerateSeqLaplacian(nx,ny,nz,v,&rhs,&x,&sol);int E=hypre_CSRMatrixNumNonzeros(A);double *dv,*du,*df;int*di,*dj;ck(cudaMalloc(&dv,E*8));ck(cudaMalloc(&di,(N+1)*4));ck(cudaMalloc(&dj,E*4));ck(cudaMalloc(&du,N*8));ck(cudaMalloc(&df,N*8));ck(cudaMemcpy(dv,hypre_CSRMatrixData(A),E*8,cudaMemcpyHostToDevice));ck(cudaMemcpy(di,hypre_CSRMatrixI(A),(N+1)*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dj,hypre_CSRMatrixJ(A),E*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(df,hypre_VectorData(sol),N*8,cudaMemcpyHostToDevice));std::vector<double>u(N),out(N);for(int i=0;i<N;i++)u[i]=1+((i*17)%31-15)/64.0;std::string dir=getenv("AMGMK_DIAGNOSTIC_DIR");for(int t=0;t<16;t++){ck(cudaMemcpy(du,u.data(),N*8,cudaMemcpyHostToDevice));relax<<<(N+255)/256,256>>>(dv,di,dj,du,df,N);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),du,N*8,cudaMemcpyDeviceToHost));std::ofstream f(dir+"/one_step_"+std::to_string(t)+"_f64.bin",std::ios::binary);f.write((char*)out.data(),N*8);}printf("SCHEDULE_DIAGNOSTIC N=%d E=%d repetitions=16 sourceblocks=%d sourcekernelunchanged=true\n",N,E,(N+255)/256);for(auto p:{dv,du,df})ck(cudaFree(p));ck(cudaFree(di));ck(cudaFree(dj));hypre_CSRMatrixDestroy(A);hypre_SeqVectorDestroy(rhs);hypre_SeqVectorDestroy(x);hypre_SeqVectorDestroy(sol);return 0;}
