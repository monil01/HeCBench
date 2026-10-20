#include <cuda_runtime.h>
#include <cublas.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstdint>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
extern "C" {
#include "hpl.h"
#include "hpl_comm.h"
#include "hpl_pmatgen.h"
#include "hpl_ptimer.h"

#include "study_api.h"
}
static hpl_gpu_callback callback=nullptr;
static FILE* output=nullptr;
static double excluded=0, max_abs=0,max_scaled=0;
static size_t checked=0,fields=0;
static int ngemm=0,ntrsm=0,ncpu_gemm=0,ncpu_trsm=0;
static auto now(){return std::chrono::steady_clock::now();}
static double sec(std::chrono::steady_clock::time_point t){return std::chrono::duration<double>(now()-t).count();}
static void die(const char*s){fprintf(stderr,"HPL study failure: %s\n",s);exit(2);}
static void ck(cudaError_t x){if(x!=cudaSuccess)die(cudaGetErrorString(x));}
static void put(double x){if(output)fwrite(&x,8,1,output);fields++;}
static void vec(const double*x,size_t n){if(output)fwrite(x,8,n,output);fields+=n;}
static void bits(const double*x,size_t n){for(size_t i=0;i<n;i++){uint64_t b;memcpy(&b,x+i,8);put(double(uint32_t(b)));put(double(uint32_t(b>>32)));}}
static void compare(double a,double b,double atol=1e-9,double rtol=1e-9){if(!std::isfinite(a)||!std::isfinite(b))die("nonfinite independent oracle");double e=fabs(a-b);max_abs=std::max(max_abs,e);max_scaled=std::max(max_scaled,e/(atol+rtol*fabs(b)));checked++;if(e>atol+rtol*fabs(b)){fprintf(stderr,"actual %.17g gold %.17g err %.17g\n",a,b,e);die("independent full vendor operation oracle");}}
// All arrays include source leading-dimension padding; readback is actual device storage.
extern "C" void hpl_observe(int kind,int m,int n,int k,double alpha,const double*A,int lda,const double*B,int ldb,double beta,double*C,int ldc,const double*dA,const double*dB,const double*dC){
 auto t=now();size_t na=size_t(lda)*(kind==0?k:m),nb=size_t(ldb)*n,nc=kind==0?size_t(ldc)*n:nb;
 std::vector<double>a(na),b(nb),c(nc);ck(cudaMemcpy(a.data(),dA,na*8,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),dB,nb*8,cudaMemcpyDeviceToHost));
 if(memcmp(a.data(),A,na*8)||memcmp(b.data(),B,nb*8))die("actual device A/B input bits differ");
 if(kind==0){ck(cudaMemcpy(c.data(),dC,nc*8,cudaMemcpyDeviceToHost));if(memcmp(c.data(),C,nc*8))die("actual device C input bits differ");}
 put(kind);put(m);put(n);put(k);put(lda);put(ldb);put(ldc);put(alpha);put(beta);vec(a.data(),na);vec(b.data(),nb);if(kind==0)vec(c.data(),nc);
 excluded+=sec(t);
}
static void vendor(int kind,int m,int n,int k,double alpha,const double*A,int lda,const double*B,int ldb,double beta,double*C,int ldc){
 auto t=now();size_t na=size_t(lda)*(kind==0?k:m),nb=size_t(ldb)*n,nc=kind==0?size_t(ldc)*n:nb;
 std::vector<double> originalA(A,A+na),originalB(B,B+nb);std::vector<double> gold(kind==0?C:B,(kind==0?C:B)+nc);
 // Deterministic owned read-only staging: normalize only mathematically inactive
 // rows copied by the original full leading-dimension spans. Host source untouched.
 for(size_t j=0;j<(kind==0?size_t(k):size_t(m));j++)for(int i=m;i<lda;i++)originalA[i+size_t(lda)*j]=0;
 if(kind==0)for(int j=0;j<n;j++)for(int i=k;i<ldb;i++)originalB[i+size_t(ldb)*j]=0;
 A=originalA.data();if(kind==0)B=originalB.data();
 excluded+=sec(t);
 if(callback)callback(kind,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
 else{
  double *a,*b,*c=nullptr;ck(cudaMalloc(&a,na*8));ck(cudaMalloc(&b,nb*8));ck(cudaMemcpy(a,A,na*8,cudaMemcpyHostToDevice));ck(cudaMemcpy(b,B,nb*8,cudaMemcpyHostToDevice));
  if(kind==0){ck(cudaMalloc(&c,nc*8));ck(cudaMemcpy(c,C,nc*8,cudaMemcpyHostToDevice));}
  ck(cudaDeviceSynchronize());hpl_observe(kind,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,a,b,c);
  if(kind==0)cublasDgemm('N','N',m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);else cublasDtrsm('L','L','N','U',m,n,alpha,a,lda,b,ldb);
  if(cublasGetError()!=CUBLAS_STATUS_SUCCESS)die("legacy cuBLAS status");ck(cudaDeviceSynchronize());ck(cudaMemcpy(kind==0?C:(double*)B,kind==0?c:b,nc*8,cudaMemcpyDeviceToHost));ck(cudaDeviceSynchronize());ck(cudaFree(a));ck(cudaFree(b));if(c)ck(cudaFree(c));
 }
 t=now();
 if(kind==0)cblas_dgemm(HplColumnMajor,HplNoTrans,HplNoTrans,m,n,k,alpha,originalA.data(),lda,originalB.data(),ldb,beta,gold.data(),ldc);else cblas_dtrsm(HplColumnMajor,HplLeft,HplLower,HplNoTrans,HplUnit,m,n,alpha,originalA.data(),lda,gold.data(),ldb);
 const double*got=kind==0?C:B;for(size_t i=0;i<nc;i++){if(fabs(got[i]-gold[i])>1e-9+1e-9*fabs(gold[i]))fprintf(stderr,"kind%d m%d n%d k%d index%zu row%zu col%zu\n",kind,m,n,k,i,i%(kind==0?ldc:ldb),i/(kind==0?ldc:ldb));compare(got[i],gold[i]);}
 vec(got,nc);excluded+=sec(t);
}
extern "C" void dpcpp_dgemm(HPL_ORDER order,HPL_TRANS ta,HPL_TRANS tb,int m,int n,int k,double a,const double*A,int lda,const double*B,int ldb,double b,double*C,int ldc){if(!m||!n||!k)return;if(n<64||m<128||k<128){ncpu_gemm++;cblas_dgemm(HplColumnMajor,HplNoTrans,HplNoTrans,m,n,k,a,A,lda,B,ldb,b,C,ldc);}else{ngemm++;vendor(0,m,n,k,a,A,lda,B,ldb,b,C,ldc);}}
extern "C" void dpcpp_dtrsm(HPL_ORDER order,HPL_SIDE side,HPL_UPLO up,HPL_TRANS trans,HPL_DIAG diag,int m,int n,double a,const double*A,int lda,double*B,int ldb){if(!m||!n)return;if(m<512||n<2*m){ncpu_trsm++;cblas_dtrsm(HplColumnMajor,HplLeft,HplLower,HplNoTrans,HplUnit,m,n,a,A,lda,B,ldb);}else{ntrsm++;vendor(1,m,n,0,a,A,lda,B,ldb,0,nullptr,ldb);}}
extern "C" int hpl_initialize(){int initialized;MPI_Initialized(&initialized);if(!initialized){int argc=0;char**argv=nullptr;MPI_Init(&argc,&argv);}int size;MPI_Comm_size(MPI_COMM_WORLD,&size);if(size!=1)die("named one-rank scope");ck(cudaFree(nullptr));if(cublasInit()!=CUBLAS_STATUS_SUCCESS)die("legacy init");return 0;}
extern "C" void hpl_finalize(){cublasShutdown();int finalized;MPI_Finalized(&finalized);if(!finalized)MPI_Finalize();}
extern "C" int hpl_run(int trial,hpl_gpu_callback cb,const char*path){
 callback=cb;output=path?fopen(path,"wb"):nullptr;if(path&&!output)die("open artifact");excluded=max_abs=max_scaled=0;checked=fields=0;ngemm=ntrsm=ncpu_gemm=ncpu_trsm=0;
 HPL_T_grid grid;HPL_grid_init(MPI_COMM_WORLD,HPL_ROW_MAJOR,1,1,&grid);
 HPL_T_palg algo{};algo.btopo=HPL_1RING;algo.depth=1;algo.nbmin=2;algo.nbdiv=2;algo.pfact=HPL_LEFT_LOOKING;algo.rfact=HPL_LEFT_LOOKING;algo.pffun=HPL_pdpanllN;algo.rffun=HPL_pdrpanllN;algo.upfun=HPL_pdupdateNN;algo.fswap=HPL_SWAP01;algo.fsthr=192;algo.equil=1;algo.align=8;
 std::vector<std::vector<double>> original;const int ns[2]={1536,257},bs[2]={512,64};
 // Exact canonical input prefix for both matrices. All source-generated coefficients retained;
 // source-generated RHS retained too, so this is original HPL matrix generation, not manufactured data.
 for(int c=0;c<2;c++){int n=ns[c],ld=(n/8+1)*8;if((ld&(ld-1))==0)ld+=8;std::vector<double> a(size_t(ld)*(n+1)+(n+1)+8,0);HPL_pdmatgen(&grid,n,n+1,bs[c],a.data(),ld,100);bits(a.data(),a.size());original.push_back(std::move(a));}
 size_t prefix=fields;double total=0, max_backward=0;
 for(int ci=0;ci<2;ci++){
  int n=ns[ci],ld=(n/8+1)*8;if((ld&(ld-1))==0)ld+=8;std::vector<double> a=original[ci],x(n+8,987654.25);
  HPL_T_pmat mat{};mat.A=a.data();mat.X=x.data();mat.n=n;mat.nb=bs[ci];mat.ld=ld;mat.mp=n;mat.nq=n+1;mat.info=0;
  put(n);put(mat.nb);put(ld);ck(cudaDeviceSynchronize());double before=excluded;auto t=now();HPL_pdgesv(&grid,&algo,&mat);ck(cudaDeviceSynchronize());total+=sec(t)-(excluded-before);if(mat.info!=0)die("HPL factorization info");
  double anorm=0,xnorm=0,bnorm=0,rnorm=0;for(int j=0;j<n;j++)xnorm=std::max(xnorm,fabs(x[j]));
  for(int i=0;i<n;i++){long double s=0,row=0;for(int j=0;j<n;j++){s+=(long double)original[ci][i+size_t(ld)*j]*x[j];row+=fabs(original[ci][i+size_t(ld)*j]);}double b=original[ci][i+size_t(ld)*n];anorm=std::max(anorm,double(row));bnorm=std::max(bnorm,fabs(b));rnorm=std::max(rnorm,double(fabsl(s-b)));}
  double backward=rnorm/(anorm*xnorm+bnorm);max_backward=std::max(max_backward,backward);if(backward>1e-12)die("independent long-double full Ax-b backward error");
  for(int j=n;j<n+8;j++)if(x[j]!=987654.25)die("solution guard");
  for(size_t j=size_t(ld)*(n+1)+(n+1);j<a.size();j++)if(a[j]!=0)die("source allocation outer guard");vec(a.data(),a.size());vec(x.data(),x.size());put(backward); // backward residual is auxiliary, not input.
 }
 HPL_grid_exit(&grid);if(output)fclose(output);output=nullptr;
 if(trial)printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_normalized_oracle_error\":%.17g,\"max_backward_error\":%.17g,\"gpu_gemm_calls\":%d,\"gpu_trsm_calls\":%d,\"cpu_gemm_calls\":%d,\"cpu_trsm_calls\":%d,\"authored_gpu_kernel_count\":0,\"excluded_observation_seconds\":%.17g,\"timing_region\":\"full_single_rank_HPL_solve_minus_explicit_observation_intervals\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"output_trial%d.bin\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"input_prefix_count\":%zu}\n",trial,total,checked,max_abs,max_scaled,max_backward,ngemm,ntrsm,ncpu_gemm,ncpu_trsm,excluded,trial,fields,prefix);
 fflush(stdout);return 0;
}
#ifndef HPL_LIBRARY
int main(){hpl_initialize();hpl_run(0,nullptr,nullptr);for(int i=1;i<=5;i++){char file[64];sprintf(file,"output_trial%d.bin",i);hpl_run(i,nullptr,file);}hpl_finalize();}
#endif
