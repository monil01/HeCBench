#pragma once
#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hipfft/hipfft.h>
#include <map>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <cstdint>

namespace tgv_dmem {
inline void check(int status){if(status){std::fprintf(stderr,"TGV IRIS/vendor failure %d\n",status);std::abort();}}
inline int integer(size_t value){if(value>INT_MAX){std::fputs("TGV dimension exceeds supported INT_MAX domain\n",stderr);std::abort();}return int(value);}
struct Storage {
 void* host;size_t bytes;std::unique_ptr<iris::DMem> mem;
 explicit Storage(size_t n):host(std::calloc(n?n:1,1)),bytes(n?n:1){if(!host)std::abort();mem=std::make_unique<iris::DMem>(host,bytes);}
 ~Storage(){mem.reset();std::free(host);}
};
inline std::map<uintptr_t,std::unique_ptr<Storage>>& storage(){static std::map<uintptr_t,std::unique_ptr<Storage>> s;return s;}
inline Storage& find(const void* p,size_t n=1){
 auto address=reinterpret_cast<uintptr_t>(p);auto it=storage().upper_bound(address);
 if(it==storage().begin())std::abort();--it;
 size_t off=address-it->first;
 if(off>=it->second->bytes||n>it->second->bytes-off)std::abort();return *it->second;
}
struct Slice{iris::DMem* mem;size_t offset;};
inline Slice slice(const void* p){auto& s=find(p);return {s.mem.get(),reinterpret_cast<uintptr_t>(p)-reinterpret_cast<uintptr_t>(s.host)};}
inline void flush(Storage& s){iris::Task t;check(t.flush_out(*s.mem));check(t.submit(0,nullptr,1));}
inline hipError_t alloc(void** p,size_t bytes){auto s=std::make_unique<Storage>(bytes);*p=s->host;storage().emplace(reinterpret_cast<uintptr_t>(*p),std::move(s));return hipSuccess;}
inline hipError_t release(void* p){if(storage().erase(reinterpret_cast<uintptr_t>(p))!=1)std::abort();return hipSuccess;}
inline hipError_t copy(void* dst,const void* src,size_t n,hipMemcpyKind mode,hipStream_t=nullptr){
 if(mode==hipMemcpyHostToDevice){auto& s=find(dst,n);if(dst!=s.host)flush(s);std::memcpy(dst,src,n);check(s.mem->update(s.host));}
 else if(mode==hipMemcpyDeviceToHost){auto& s=find(src,n);flush(s);std::memcpy(dst,src,n);}
 else if(mode==hipMemcpyDeviceToDevice){
  find(src,n);find(dst,n);auto a=slice(dst),b=slice(src);
  void* p[]={a.mem,b.mem,&n,&a.offset,&b.offset};int pi[]={iris_rw,iris_r,sizeof(n),sizeof(a.offset),sizeof(b.offset)};
  size_t g[]={256},l[]={256};iris::Task t;check(t.kernel("copy_bytes",1,nullptr,g,l,5,p,pi));check(t.submit(0,nullptr,1));
 }else std::abort();return hipSuccess;
}
inline hipError_t zero(void* dst,int value,size_t n){
 if(value!=0)std::abort();find(dst,n);auto a=slice(dst);
 void* p[]={a.mem,&n,&a.offset};int pi[]={iris_rw,sizeof(n),sizeof(a.offset)};size_t g[]={256},l[]={256};
 iris::Task t;check(t.kernel("zero_bytes",1,nullptr,g,l,3,p,pi));check(t.submit(0,nullptr,1));return hipSuccess;
}
enum {IMAX=1,NRM2,FFT_SETUP,FFT_EXEC,SVD_SIZE,SVD,DGMM,GEMM,DESTROY_BLAS,DESTROY_SOLVER,FINISH};
struct Shape{int m=0,n=0,k=0,lda=0,ldb=0,ldc=0,lwork=0,dir=0;};
static_assert(sizeof(Shape)==32,"bridge descriptor size");
struct Context{
 hipblasHandle_t blas=nullptr;hipsolverHandle_t solver=nullptr;hipfftHandle fft=0;bool fft_ready=false;
 size_t fft_bytes=0;void* fft_token=nullptr;int lwork=0;
 size_t forward=0,backward=0,svd=0,dgmm=0,gemm=0;
};
inline Context& context(){static Context c;return c;}
inline int callback(void* stream,unsigned,void*const* args){
 auto* c=hecbench_vendor_arg<Context*>(args,1);int op=hecbench_vendor_arg<int>(args,2);
 auto s=hecbench_vendor_arg<Shape>(args,3);auto hs=static_cast<hipStream_t>(stream);
 if(op==DESTROY_BLAS){if(c->blas){auto status=hipblasDestroy(c->blas);c->blas=nullptr;return int(status);}return 0;}
 if(op==DESTROY_SOLVER){if(c->solver){auto status=hipsolverDnDestroy(c->solver);c->solver=nullptr;return int(status);}return 0;}
 if(op==FINISH){if(c->fft_ready){c->fft_ready=false;return int(hipfftDestroy(c->fft));}return 0;}
 if(op==FFT_SETUP){
  int n[]={s.n,s.m};
  if(hipfftCreate(&c->fft)!=HIPFFT_SUCCESS)return 100;
  if(hipfftSetAutoAllocation(c->fft,0)!=HIPFFT_SUCCESS)return 101;
  auto result=hipfftMakePlanMany(c->fft,2,n,nullptr,s.k,1,nullptr,s.k,1,HIPFFT_C2C,s.k,&c->fft_bytes);
  if(result!=HIPFFT_SUCCESS)return 102+int(result);c->fft_ready=true;return 0;
 }
 if(op==FFT_EXEC){
  auto in=hecbench_vendor_arg<hipfftComplex*>(args,4),out=hecbench_vendor_arg<hipfftComplex*>(args,5);
  auto work=hecbench_vendor_arg<void*>(args,6);
  if(hipfftSetStream(c->fft,hs)!=HIPFFT_SUCCESS)return 110;
  if(hipfftSetWorkArea(c->fft,work)!=HIPFFT_SUCCESS)return 111;
  auto result=hipfftExecC2C(c->fft,in,out,s.dir);if(result!=HIPFFT_SUCCESS)return 112+int(result);
  if(s.dir==HIPFFT_FORWARD)++c->forward;else ++c->backward;
  return int(hipStreamSynchronize(hs));
 }
 if(op==SVD_SIZE||op==SVD){
  if(!c->solver&&hipsolverDnCreate(&c->solver)!=HIPSOLVER_STATUS_SUCCESS)return 120;
  if(hipsolverDnSetStream(c->solver,hs)!=HIPSOLVER_STATUS_SUCCESS)return 121;
  if(op==SVD_SIZE)return int(hipsolverDnCgesvd_bufferSize(c->solver,s.m,s.n,&c->lwork));
  auto a=hecbench_vendor_arg<hipFloatComplex*>(args,4);
  auto singular=hecbench_vendor_arg<float*>(args,5);
  auto u=hecbench_vendor_arg<hipFloatComplex*>(args,6),vt=hecbench_vendor_arg<hipFloatComplex*>(args,7);
  auto work=hecbench_vendor_arg<hipFloatComplex*>(args,8);
  auto rwork=hecbench_vendor_arg<float*>(args,9);auto info=hecbench_vendor_arg<int*>(args,10);
  auto result=hipsolverDnCgesvd(c->solver,'S','S',s.m,s.n,a,s.lda,singular,u,s.ldb,vt,s.ldc,work,s.lwork,rwork,info);
  if(result!=HIPSOLVER_STATUS_SUCCESS)return 122+int(result);++c->svd;return int(hipStreamSynchronize(hs));
 }
 if(!c->blas&&hipblasCreate(&c->blas)!=HIPBLAS_STATUS_SUCCESS)return 130;
 if(hipblasSetStream(c->blas,hs)!=HIPBLAS_STATUS_SUCCESS)return 131;
 auto a=hecbench_vendor_arg<hipComplex*>(args,4);auto b=hecbench_vendor_arg<hipComplex*>(args,5);
 hipblasStatus_t result=HIPBLAS_STATUS_INVALID_VALUE;
 if(op==IMAX||op==NRM2){
  if(hipblasSetPointerMode(c->blas,HIPBLAS_POINTER_MODE_DEVICE)!=HIPBLAS_STATUS_SUCCESS)return 132;
  result=op==IMAX?hipblasIcamax_64(c->blas,s.m,a,s.lda,reinterpret_cast<int64_t*>(b)):
   hipblasSnrm2_64(c->blas,s.m,reinterpret_cast<float*>(a),s.lda,reinterpret_cast<float*>(b));
  if(hipblasSetPointerMode(c->blas,HIPBLAS_POINTER_MODE_HOST)!=HIPBLAS_STATUS_SUCCESS)return 133;
 }
 if(op==DGMM){auto out=hecbench_vendor_arg<hipComplex*>(args,6);result=hipblasCdgmm_64(c->blas,HIPBLAS_SIDE_LEFT,s.m,s.n,a,s.lda,b,s.ldb,out,s.ldc);if(result==HIPBLAS_STATUS_SUCCESS)++c->dgmm;}
 if(op==GEMM){
  auto out=hecbench_vendor_arg<hipComplex*>(args,6);
  auto alpha=hecbench_vendor_arg<hipComplex>(args,7),beta=hecbench_vendor_arg<hipComplex>(args,8);
  result=hipblasCgemm_64(c->blas,HIPBLAS_OP_N,HIPBLAS_OP_N,s.m,s.n,s.k,&alpha,a,s.lda,b,s.ldb,&beta,out,s.ldc);
  if(result==HIPBLAS_STATUS_SUCCESS)++c->gemm;
 }
 return result==HIPBLAS_STATUS_SUCCESS?int(hipStreamSynchronize(hs)):140+int(result);
}
inline int submit(int op,Shape s,std::vector<Slice> arrays={},int flush_index=-1,hipComplex alpha={},hipComplex beta={}){
 auto* c=&context();hecbench_vendor_callback cb=callback;
 std::vector<void*> p{&cb,&c,&op,&s};std::vector<int> pi{int(sizeof(cb)),int(sizeof(c)),int(sizeof(op)),int(sizeof(s))};
 for(auto a:arrays){if(a.offset)std::abort();p.push_back(a.mem);pi.push_back(iris_rw);}
 if(op==GEMM){p.push_back(&alpha);p.push_back(&beta);pi.push_back(sizeof(alpha));pi.push_back(sizeof(beta));}
 // Accurate access modes; workspaces and in-place operands remain read/write.
 if(op==IMAX||op==NRM2){pi[4]=iris_r;pi[5]=iris_w;}
 if(op==DGMM||op==GEMM){pi[4]=iris_r;pi[5]=iris_r;}
 if(op==FFT_EXEC&&arrays[0].mem!=arrays[1].mem)pi[4]=iris_r;
 size_t g[]={1},l[]={1};iris::Task t;check(t.kernel("hecbench_vendor",1,nullptr,g,l,p.size(),p.data(),pi.data()));
 if(flush_index>=0)check(t.flush_out(*arrays.at(flush_index).mem));check(t.submit(0,nullptr,1));return 0;
}
inline void finish(){auto& c=context();submit(FINISH,{});if(c.fft_token){release(c.fft_token);c.fft_token=nullptr;}
 std::printf("IRIS actual vendor calls FFT_forward=%zu FFT_backward=%zu SVD=%zu Cdgmm_64=%zu Cgemm_64=%zu\n",c.forward,c.backward,c.svd,c.dgmm,c.gemm);
 if(!storage().empty()){std::fputs("Unreleased TGV DMem storage\n",stderr);std::abort();}
}
}

inline hipblasStatus_t tgv_blas_create(int* h){*h=1;return HIPBLAS_STATUS_SUCCESS;}
inline hipsolverStatus_t tgv_solver_create(int* h){*h=1;return HIPSOLVER_STATUS_SUCCESS;}
inline hipblasStatus_t tgv_blas_destroy(int){tgv_dmem::submit(tgv_dmem::DESTROY_BLAS,{});return HIPBLAS_STATUS_SUCCESS;}
inline hipsolverStatus_t tgv_solver_destroy(int){tgv_dmem::submit(tgv_dmem::DESTROY_SOLVER,{});return HIPSOLVER_STATUS_SUCCESS;}
inline hipblasStatus_t tgv_imax(int,size_t n,const hipComplex* a,int inc,long* out){
 int64_t result=-1;iris::DMem mem(&result,sizeof(result));tgv_dmem::Shape s;s.m=tgv_dmem::integer(n);s.lda=inc;
 tgv_dmem::submit(tgv_dmem::IMAX,s,{tgv_dmem::slice(a),{&mem,0}},1);*out=result;return HIPBLAS_STATUS_SUCCESS;
}
inline hipblasStatus_t tgv_nrm2(int,size_t n,const float* a,int inc,float* out){
 *out=0;iris::DMem mem(out,sizeof(*out));tgv_dmem::Shape s;s.m=tgv_dmem::integer(n);s.lda=inc;
 tgv_dmem::submit(tgv_dmem::NRM2,s,{tgv_dmem::slice(a),{&mem,0}},1);return HIPBLAS_STATUS_SUCCESS;
}
inline hipfftResult tgv_fft_plan(int* plan,int rank,int* n,int*,int stride,int dist,int*,int ostride,int odist,hipfftType type,int batch){
 if(rank!=2||stride!=batch||ostride!=batch||dist!=1||odist!=1||type!=HIPFFT_C2C)std::abort();
 tgv_dmem::Shape s;s.m=n[1];s.n=n[0];s.k=batch;tgv_dmem::submit(tgv_dmem::FFT_SETUP,s);
 auto& c=tgv_dmem::context();tgv_dmem::alloc(&c.fft_token,c.fft_bytes);*plan=1;return HIPFFT_SUCCESS;
}
inline hipfftResult tgv_fft_exec(int,hipfftComplex* a,hipfftComplex* b,int dir){
 tgv_dmem::Shape s;s.dir=dir;tgv_dmem::submit(tgv_dmem::FFT_EXEC,s,{tgv_dmem::slice(a),tgv_dmem::slice(b),tgv_dmem::slice(tgv_dmem::context().fft_token)});return HIPFFT_SUCCESS;
}
inline hipsolverStatus_t tgv_svd_size(int,int m,int n,int* out){
 tgv_dmem::Shape s;s.m=m;s.n=n;tgv_dmem::submit(tgv_dmem::SVD_SIZE,s);*out=tgv_dmem::context().lwork;return HIPSOLVER_STATUS_SUCCESS;
}
inline hipsolverStatus_t tgv_svd(int,signed char ju,signed char jv,int m,int n,hipFloatComplex* a,int lda,float* values,hipFloatComplex* u,int ldu,hipFloatComplex* vt,int ldvt,hipFloatComplex* work,int lwork,float* rwork,int* info){
 if(ju!='S'||jv!='S')std::abort();tgv_dmem::Shape s;s.m=m;s.n=n;s.lda=lda;s.ldb=ldu;s.ldc=ldvt;s.lwork=lwork;
 tgv_dmem::submit(tgv_dmem::SVD,s,{tgv_dmem::slice(a),tgv_dmem::slice(values),tgv_dmem::slice(u),tgv_dmem::slice(vt),tgv_dmem::slice(work),tgv_dmem::slice(rwork),tgv_dmem::slice(info)},6);
 if(*info){std::fprintf(stderr,"Complex SVD devInfo=%d\n",*info);std::abort();}return HIPSOLVER_STATUS_SUCCESS;
}
inline hipblasStatus_t tgv_dgmm(int,hipblasSideMode_t side,int64_t m,int64_t n,const hipComplex* a,int64_t lda,const hipComplex* x,int64_t inc,hipComplex* out,int64_t ldc){
 if(side!=HIPBLAS_SIDE_LEFT)std::abort();tgv_dmem::Shape s;s.m=tgv_dmem::integer(m);s.n=tgv_dmem::integer(n);s.lda=tgv_dmem::integer(lda);s.ldb=tgv_dmem::integer(inc);s.ldc=tgv_dmem::integer(ldc);
 tgv_dmem::submit(tgv_dmem::DGMM,s,{tgv_dmem::slice(a),tgv_dmem::slice(x),tgv_dmem::slice(out)});return HIPBLAS_STATUS_SUCCESS;
}
inline hipblasStatus_t tgv_gemm(int,hipblasOperation_t ta,hipblasOperation_t tb,int64_t m,int64_t n,int64_t k,const hipComplex* alpha,const hipComplex* a,int64_t lda,const hipComplex* b,int64_t ldb,const hipComplex* beta,hipComplex* out,int64_t ldc){
 if(ta!=HIPBLAS_OP_N||tb!=HIPBLAS_OP_N)std::abort();tgv_dmem::Shape s;s.m=tgv_dmem::integer(m);s.n=tgv_dmem::integer(n);s.k=tgv_dmem::integer(k);s.lda=tgv_dmem::integer(lda);s.ldb=tgv_dmem::integer(ldb);s.ldc=tgv_dmem::integer(ldc);
 tgv_dmem::submit(tgv_dmem::GEMM,s,{tgv_dmem::slice(a),tgv_dmem::slice(b),tgv_dmem::slice(out)},-1,*alpha,*beta);return HIPBLAS_STATUS_SUCCESS;
}
