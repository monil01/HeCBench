// Original FRNA kernels/host lifecycle with diagnostic allocation labels only.
#include <cuda_runtime.h>
#include <cstdio>
static int allocation_number=0;
template<class T>cudaError_t traced_allocate(T**p,size_t bytes){auto e=cudaMalloc(p,bytes);const char*names[]={"V","W","WM","W5base","W3","WCA","parameters","sequence"};fprintf(stderr,"FRNA_ALLOCATION %d %s %p %zu\n",allocation_number,names[allocation_number%8],(void*)*p,bytes);allocation_number++;return e;}
#define cudaMalloc traced_allocate
#include "configured_native/frna.cu"
#undef cudaMalloc
int main(int argc,char**argv){if(argc!=3)return 2;fparam par;fparam_read_from_text(argv[1],0,&par);char*seq=fsequence(argv[2]);frna_t p=frna_new(seq,&par);frna_delete(p);free(seq);}
