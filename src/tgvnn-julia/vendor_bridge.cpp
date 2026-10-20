// Host-only CUDA runtime / cuFFT / cuBLAS / cuSOLVER interoperability.
// No authored device code, PTX or CUDA kernel launch is present in this library.
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <utility>
#include <initializer_list>
struct Ctx{cublasHandle_t b;cusolverDnHandle_t s;cufftHandle f;int work;cuComplex *u,*s2,*vt,*svt,*w;float *sing,*rw;int *info;};
#define OK(x) do{int e=int(x);if(e){fprintf(stderr,"VENDOR_FAILURE %s=%d\n",#x,e);abort();}}while(0)
extern "C" {
void*tgv_alloc(size_t n){void*p;OK(cudaMalloc(&p,n));return p;}
void tgv_free(void*p){OK(cudaFree(p));}
void tgv_h2d(void*d,const void*s,size_t n){OK(cudaMemcpyAsync(d,s,n,cudaMemcpyHostToDevice,0));}
void tgv_d2d(void*d,const void*s,size_t n){OK(cudaMemcpyAsync(d,s,n,cudaMemcpyDeviceToDevice,0));}
void tgv_zero(void*d,size_t n){OK(cudaMemset(d,0,n));}
float tgv_alpha(float a0,float a1,int iter,int index){return expf(float(index)/float(iter)*logf(a1)+(float(iter)-float(index))/float(iter)*logf(a0));}
void*tgv_create(){Ctx*c=new Ctx{};OK(cublasCreate(&c->b));OK(cusolverDnCreate(&c->s));return c;}
void tgv_fft_init(Ctx*c,int r,int cols,int nt){int dims[2]={cols,r};OK(cufftPlanMany(&c->f,2,dims,nullptr,nt,1,nullptr,nt,1,CUFFT_C2C,nt));}
void tgv_fft(Ctx*c,cuComplex*out,cuComplex*in,int direction){OK(cufftExecC2C(c->f,in,out,direction));}
float tgv_normalizer(Ctx*c,cuComplex*a,int64_t n){int64_t index;OK(cublasIcamax_64(c->b,n,a,1,&index));cuComplex h;OK(cudaMemcpyAsync(&h,a+index-1,sizeof h,cudaMemcpyDeviceToHost,0));return std::abs(h.x);}
void tgv_svd_init(Ctx*c,int m,int n){OK(cusolverDnCgesvd_bufferSize(c->s,m,n,&c->work));c->u=(cuComplex*)tgv_alloc(size_t(m)*n*sizeof(cuComplex));c->sing=(float*)tgv_alloc(n*sizeof(float));c->s2=(cuComplex*)tgv_alloc(n*sizeof(cuComplex));c->vt=(cuComplex*)tgv_alloc(size_t(n)*n*sizeof(cuComplex));c->svt=(cuComplex*)tgv_alloc(size_t(n)*n*sizeof(cuComplex));c->w=(cuComplex*)tgv_alloc(c->work*sizeof(cuComplex));c->rw=(float*)tgv_alloc((n-1)*sizeof(float));c->info=(int*)tgv_alloc(sizeof(int));}
void*tgv_buffer(Ctx*c,int i){switch(i){case 0:return c->u;case 1:return c->sing;case 2:return c->s2;case 3:return c->vt;case 4:return c->svt;case 5:return c->info;default:return nullptr;}}
void tgv_svd(Ctx*c,cuComplex*A,int m,int n){OK(cusolverDnCgesvd(c->s,'S','S',m,n,A,m,c->sing,c->u,m,c->vt,n,c->w,c->work,c->rw,c->info));}
void tgv_reconstruct(Ctx*c,cuComplex*A,int m,int n){cuComplex a=make_cuComplex(1,0),b=make_cuComplex(0,0);OK(cublasCdgmm_64(c->b,CUBLAS_SIDE_LEFT,n,n,c->vt,n,c->s2,1,c->svt,n));OK(cublasCgemm_64(c->b,CUBLAS_OP_N,CUBLAS_OP_N,m,n,n,&a,c->u,m,c->svt,n,&b,A,m));}
float tgv_norm(Ctx*c,float*a,int64_t n){float v;OK(cublasSnrm2_64(c->b,n,a,1,&v));return v;}
void tgv_source_destroy(Ctx*c){for(void*p:{(void*)c->u,(void*)c->sing,(void*)c->s2,(void*)c->vt,(void*)c->svt,(void*)c->w,(void*)c->rw,(void*)c->info})tgv_free(p);OK(cublasDestroy(c->b));OK(cusolverDnDestroy(c->s));}
void tgv_finish_destroy(Ctx*c){OK(cufftDestroy(c->f));delete c;}
void tgv_resolution(){for(auto z:{std::pair<const char*,void*>{"cuFFT",(void*)&cufftExecC2C},{"cuBLAS",(void*)&cublasCgemm_64},{"cuSOLVER",(void*)&cusolverDnCgesvd}}){Dl_info d;if(!dladdr(z.second,&d))abort();printf("VENDOR_RESOLUTION %s %s\n",z.first,d.dli_fname);}fflush(stdout);}
}
