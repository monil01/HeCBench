#pragma once
#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <map>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
namespace rbc_dmem {
struct Storage {
 void* host;size_t bytes;std::unique_ptr<iris::DMem> mem;
 explicit Storage(size_t n):host(std::malloc(n)),bytes(n),mem(new iris::DMem(host,n)){if(!host)std::abort();}
 ~Storage(){mem.reset();std::free(host);}
};
inline std::map<uintptr_t,std::unique_ptr<Storage>>& storage(){static std::map<uintptr_t,std::unique_ptr<Storage>> s;return s;}
struct Slice {iris::DMem* mem;size_t offset;};
inline Storage& find(const void* p){
 auto it=storage().upper_bound(reinterpret_cast<uintptr_t>(p));
 if(it==storage().begin())std::abort();--it;
 if(reinterpret_cast<uintptr_t>(p)-it->first>=it->second->bytes)std::abort();return *it->second;
}
inline Slice slice(const void* p){auto& s=find(p);return {s.mem.get(),(reinterpret_cast<uintptr_t>(p)-reinterpret_cast<uintptr_t>(s.host))/sizeof(float)};}
inline void checked(int status){if(status!=IRIS_SUCCESS){fprintf(stderr,"IRIS DMem operation failed: %d\n",status);std::abort();}}
inline void flush(Storage& s){iris::Task t;checked(t.flush_out(*s.mem));checked(t.submit(0,nullptr,1));}
inline int alloc(void** p,size_t bytes){auto s=std::make_unique<Storage>(bytes);*p=s->host;storage().emplace(reinterpret_cast<uintptr_t>(*p),std::move(s));return 0;}
inline int release(void* p){if(storage().erase(reinterpret_cast<uintptr_t>(p))!=1)std::abort();return 0;}
inline int setvector(int n,int bytes,const void* src,int incx,void* dst,int incy){
 auto& s=find(dst);
 for(int i=0;i<n;++i)std::memcpy(static_cast<char*>(dst)+size_t(i)*incy*bytes,static_cast<const char*>(src)+size_t(i)*incx*bytes,bytes);
 checked(s.mem->update(s.host));return 0;
}
inline int getvector(int n,int bytes,const void* src,int incx,void* dst,int incy){
 auto& s=find(src);flush(s);
 for(int i=0;i<n;++i)std::memcpy(static_cast<char*>(dst)+size_t(i)*incy*bytes,static_cast<const char*>(src)+size_t(i)*incx*bytes,bytes);return 0;
}
enum {COPY=1,SCALE,AXPY,DOT,IAMAX,GEMM,DESTROY};
struct Shape {int n=0,m=0,k=0,lda=0,ldb=0,ldc=0,incx=1,incy=1;};
static_assert(sizeof(Shape)==32,"bridge scalar descriptor");
inline int callback(void* stream,unsigned,void*const* args){
 static thread_local hipblasHandle_t h=nullptr;
 int op=hecbench_vendor_arg<int>(args,1);
 if(op==DESTROY){int status=h?int(hipblasDestroy(h)):0;h=nullptr;return status;}
 if(!h&&hipblasCreate(&h)!=HIPBLAS_STATUS_SUCCESS)return 1;
 if(hipblasSetStream(h,static_cast<hipStream_t>(stream))!=HIPBLAS_STATUS_SUCCESS)return 2;
 auto s=hecbench_vendor_arg<Shape>(args,2);
 float alpha=hecbench_vendor_arg<float>(args,3),beta=hecbench_vendor_arg<float>(args,4);
 auto a=hecbench_vendor_arg<float*>(args,5)+hecbench_vendor_arg<size_t>(args,8);
 auto b=hecbench_vendor_arg<float*>(args,6)+hecbench_vendor_arg<size_t>(args,9);
 auto c=hecbench_vendor_arg<float*>(args,7)+hecbench_vendor_arg<size_t>(args,10);
 hipblasStatus_t status=HIPBLAS_STATUS_INVALID_VALUE;
 if(op==COPY)status=hipblasScopy(h,s.n,a,s.incx,b,s.incy);
 if(op==SCALE)status=hipblasSscal(h,s.n,&alpha,a,s.incx);
 if(op==AXPY)status=hipblasSaxpy(h,s.n,&alpha,a,s.incx,b,s.incy);
 if(op==GEMM)status=hipblasSgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,s.m,s.n,s.k,&alpha,a,s.lda,b,s.ldb,&beta,c,s.ldc);
 if(op==DOT||op==IAMAX){
  if(hipblasSetPointerMode(h,HIPBLAS_POINTER_MODE_DEVICE)!=HIPBLAS_STATUS_SUCCESS)return 3;
  status=op==DOT?hipblasSdot(h,s.n,a,s.incx,b,s.incy,c):hipblasIsamax(h,s.n,a,s.incx,reinterpret_cast<int*>(c));
  if(hipblasSetPointerMode(h,HIPBLAS_POINTER_MODE_HOST)!=HIPBLAS_STATUS_SUCCESS)return 4;
 }
 return status==HIPBLAS_STATUS_SUCCESS?int(hipStreamSynchronize(static_cast<hipStream_t>(stream))):int(status);
}
inline int submit(int op,Shape shape,Slice a,Slice b,Slice c,float alpha=0,float beta=0,bool fetch=false){
 hecbench_vendor_callback cb=callback;
 void* p[]={&cb,&op,&shape,&alpha,&beta,a.mem,b.mem,c.mem,&a.offset,&b.offset,&c.offset};
 int pi[]={sizeof(cb),sizeof(op),sizeof(shape),sizeof(alpha),sizeof(beta),iris_r,iris_r,iris_r,sizeof(a.offset),sizeof(b.offset),sizeof(c.offset)};
 if(op==COPY||op==AXPY)pi[6]=iris_rw;
 if(op==SCALE)pi[5]=iris_rw;
 if(op==GEMM)pi[7]=iris_rw;
 if(op==DOT||op==IAMAX)pi[7]=iris_w;
 size_t g[]={1},l[]={1};iris::Task t;checked(t.kernel("hecbench_vendor",1,nullptr,g,l,11,p,pi));if(fetch)checked(t.flush_out(*c.mem));checked(t.submit(0,nullptr,1));return HIPBLAS_STATUS_SUCCESS;
}
}
using DmemBlasHandle=int;
inline hipblasStatus_t dmem_create(DmemBlasHandle* h){*h=1;return HIPBLAS_STATUS_SUCCESS;}
inline int dmem_destroy(DmemBlasHandle){
 int op=rbc_dmem::DESTROY;hecbench_vendor_callback cb=rbc_dmem::callback;void* p[]={&cb,&op};int pi[]={sizeof(cb),sizeof(op)};size_t g[]={1},l[]={1};iris::Task t;rbc_dmem::checked(t.kernel("hecbench_vendor",1,nullptr,g,l,2,p,pi));rbc_dmem::checked(t.submit(0,nullptr,1));return 0;
}
inline int dmem_copy(DmemBlasHandle,int n,const float*x,int ix,float*y,int iy){rbc_dmem::Shape s;s.n=n;s.incx=ix;s.incy=iy;auto a=rbc_dmem::slice(x),b=rbc_dmem::slice(y);return rbc_dmem::submit(rbc_dmem::COPY,s,a,b,b);}
inline int dmem_scale(DmemBlasHandle,int n,const float* alpha,float*x,int ix){rbc_dmem::Shape s;s.n=n;s.incx=ix;auto a=rbc_dmem::slice(x);return rbc_dmem::submit(rbc_dmem::SCALE,s,a,a,a,*alpha);}
inline int dmem_axpy(DmemBlasHandle,int n,const float*alpha,const float*x,int ix,float*y,int iy){rbc_dmem::Shape s;s.n=n;s.incx=ix;s.incy=iy;auto a=rbc_dmem::slice(x),b=rbc_dmem::slice(y);return rbc_dmem::submit(rbc_dmem::AXPY,s,a,b,b,*alpha);}
inline int dmem_gemm(DmemBlasHandle,hipblasOperation_t ta,hipblasOperation_t tb,int m,int n,int k,const float*alpha,const float*a,int lda,const float*b,int ldb,const float*beta,float*c,int ldc){
 if(ta!=HIPBLAS_OP_N||tb!=HIPBLAS_OP_N)std::abort();rbc_dmem::Shape s;s.m=m;s.n=n;s.k=k;s.lda=lda;s.ldb=ldb;s.ldc=ldc;
 return rbc_dmem::submit(rbc_dmem::GEMM,s,rbc_dmem::slice(a),rbc_dmem::slice(b),rbc_dmem::slice(c),*alpha,*beta);
}
inline int dmem_dot(DmemBlasHandle,int n,const float*a,int ia,const float*b,int ib,float*out){
 rbc_dmem::Shape s;s.n=n;s.incx=ia;s.incy=ib;iris::DMem result(out,sizeof(*out));return rbc_dmem::submit(rbc_dmem::DOT,s,rbc_dmem::slice(a),rbc_dmem::slice(b),{&result,0},0,0,true);
}
inline int dmem_imax(DmemBlasHandle,int n,const float*a,int ia,int*out){
 rbc_dmem::Shape s;s.n=n;s.incx=ia;iris::DMem result(out,sizeof(*out));auto x=rbc_dmem::slice(a);return rbc_dmem::submit(rbc_dmem::IAMAX,s,x,x,{&result,0},0,0,true);
}
inline void launch_multiply(const char* name,dim3 grid,dim3 block,float*a,const float*b){
 auto x=rbc_dmem::slice(a),y=rbc_dmem::slice(b);void* p[]={x.mem,y.mem,&x.offset,&y.offset};int pi[]={iris_rw,iris_r,sizeof(x.offset),sizeof(y.offset)};
 size_t g[]={size_t(grid.x)*block.x,size_t(grid.y)*block.y,size_t(grid.z)*block.z},l[]={block.x,block.y,block.z};iris::Task t;rbc_dmem::checked(t.kernel(name,3,nullptr,g,l,4,p,pi));rbc_dmem::checked(t.submit(0,nullptr,1));
}
inline void launch_ElemMultOmega(dim3 g,dim3 l,float*a,const float*b){launch_multiply("ElemMultOmega",g,l,a,b);}
inline void launch_ElemMultT(dim3 g,dim3 l,float*a,const float*b){launch_multiply("ElemMultT",g,l,a,b);}
inline void launch_ElemMultNu(dim3 g,dim3 l,float*a,const float*b){launch_multiply("ElemMultNu",g,l,a,b);}
inline void launch_unary(const char* name,dim3 grid,dim3 block,float*a,float value=0){
 auto x=rbc_dmem::slice(a);void* p[]={x.mem,&value,&x.offset};int pi[]={iris_rw,sizeof(value),sizeof(x.offset)};
 size_t g[]={size_t(grid.x)*block.x,size_t(grid.y)*block.y,size_t(grid.z)*block.z},l[]={block.x,block.y,block.z};iris::Task t;rbc_dmem::checked(t.kernel(name,3,nullptr,g,l,3,p,pi));rbc_dmem::checked(t.submit(0,nullptr,1));
}
inline void launch_SubOne(dim3 g,dim3 l,float*a){launch_unary("SubOne",g,l,a);}
inline void launch_AddOne(dim3 g,dim3 l,float*a){launch_unary("AddOne",g,l,a);}
inline void launch_AddX(dim3 g,dim3 l,float*a,float x){launch_unary("AddX",g,l,a,x);}
inline void launch_Updatedt(dim3 grid,dim3 block,int iu,const float*u,int iv,const float*v,float*dt){
 auto a=rbc_dmem::slice(u),b=rbc_dmem::slice(v),c=rbc_dmem::slice(dt);
 void* p[]={&iu,a.mem,&iv,b.mem,c.mem,&a.offset,&b.offset,&c.offset};int pi[]={sizeof(iu),iris_r,sizeof(iv),iris_r,iris_w,sizeof(a.offset),sizeof(b.offset),sizeof(c.offset)};
 size_t g[]={size_t(grid.x)*block.x,size_t(grid.y)*block.y,size_t(grid.z)*block.z},l[]={block.x,block.y,block.z};iris::Task t;rbc_dmem::checked(t.kernel("Updatedt",3,nullptr,g,l,8,p,pi));rbc_dmem::checked(t.submit(0,nullptr,1));
}
