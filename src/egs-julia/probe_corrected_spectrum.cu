// Corrected original CPU alias preparation; scientific marginal probability witness.
#define CUDA_EGS
#include "configured_native/EGS.h"
#ifdef STUDY_HOST_ONLY
#include <cstring>
template<class T>cudaError_t host_allocate(T**p,size_t n){*p=static_cast<T*>(malloc(n));return *p?cudaSuccess:cudaErrorMemoryAllocation;}
static cudaError_t host_free(void*p){free(p);return cudaSuccess;}
#define cudaMalloc host_allocate
#define cudaFree host_free
#define cudaMemcpy(d,s,n,k) (memcpy(d,s,n),cudaSuccess)
#define cudaMemcpyToSymbol(...) cudaSuccess
#define cudaGetLastError() cudaSuccess
#endif
#include "configured_native/media.c"
#include "configured_native/init.cu"
#include <vector>
#include <cassert>
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDAerror %d %s\n",int(e),cudaGetErrorString(e));exit(3);}}
int main(int argc,char**argv){assert(argc==2);setvbuf(stdout,nullptr,_IOLBF,0);spec_file=argv[1];init_source();ck(cudaGetLastError());assert(h_source.n==145);std::vector<float>x(146),w(146);std::vector<int>b(146);ck(cudaMemcpy(x.data(),h_source.xi,146*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(w.data(),h_source.wi,146*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),h_source.bin,146*4,cudaMemcpyDeviceToHost));const char*names[]={"spectrum_xi.bin","spectrum_wi.bin","spectrum_alias.bin"};const void*ptrs[]={x.data(),w.data(),b.data()};for(int i=0;i<3;i++){FILE*f=fopen(names[i],"wb");assert(f);assert(fwrite(ptrs[i],4,146,f)==146);fclose(f);}printf("SPECTRUM_COLUMNS %u EDGES 146\n",h_source.n);ck(cudaFree(h_source.xi));ck(cudaFree(h_source.wi));ck(cudaFree(h_source.bin));}
