#pragma once
#include "study_api.h"
cudaError_t study_malloc_raw(void**,size_t);
cudaError_t study_free(void*);
template<class T>cudaError_t study_malloc(T**p,size_t n){return study_malloc_raw((void**)p,n);}
#define cudaMalloc study_malloc
#define cudaFree study_free
template<class T>void study_dot(int n,const T*x,const T*y,T*z){LCStage s{};s.stage=0;s.n=n;s.blocks=(n+255)/256;s.p[0]=(void*)x;s.p[1]=(void*)y;s.p[2]=z;if(study_call(&s))exit(3);}
template<class T>void study_scale(int n,T*x,T a){LCStage s{};s.stage=1;s.n=n;s.blocks=(n+255)/256;s.scalar=a;s.p[0]=x;if(study_call(&s))exit(3);}
template<class T>void study_axpy(int n,T*y,const T*x,T a){LCStage s{};s.stage=2;s.n=n;s.blocks=(n+255)/256;s.scalar=a;s.p[0]=y;s.p[1]=(void*)x;if(study_call(&s))exit(3);}
template<class T>void study_spmv(int g,int n,int nz,const int*rp,const int*ci,const T*v,const T*x,T*y){LCStage s{};s.stage=3;s.n=n;s.nnz=nz;s.group=g;s.blocks=(n+(256/g)-1)/(256/g);s.p[0]=(void*)rp;s.p[1]=(void*)ci;s.p[2]=(void*)v;s.p[3]=(void*)x;s.p[4]=y;if(study_call(&s))exit(3);}
