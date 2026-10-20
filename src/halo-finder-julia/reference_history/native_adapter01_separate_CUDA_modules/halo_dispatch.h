#pragma once
// Owned observer/dispatch adapter declarations; original scientific kernel intact.
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct HaloArgs{
 int count,count1,gx,gy,bx,by;
 const float *x,*y,*z,*mass,*x1,*y1,*z1,*mass1;
 float *vx,*vy,*vz;
 float cut2,soft2,fcoeff;
 void *stream;
};
using HaloCallback=int(*)(const HaloArgs*);
using HaloObserver=void(*)(int,const HaloArgs*);
struct HaloAllocation{void*base;void*pointer;size_t bytes;};
extern std::map<void*,HaloAllocation>halo_allocations;
extern HaloCallback halo_target_callback;
extern HaloObserver halo_observer_callback;
static constexpr uint32_t HALO_GUARD=0xa5a5a5a5;
static inline void halo_check_guards(void*p){auto a=halo_allocations.at(p);auto*q=(uint32_t*)a.base;for(size_t i=0;i<32;i++)if(q[i]!=HALO_GUARD||q[32+a.bytes/4+i]!=HALO_GUARD){fprintf(stderr,"HALO_HOSTPINNED_GUARD_FAILED\n");exit(7);}}
template<class T>cudaError_t halo_malloc_host(T**p,size_t bytes){
 void*base;cudaError_t e=cudaMallocHost(&base,bytes+256);if(e!=cudaSuccess)return e;memset(base,0xa5,bytes+256);void*payload=(char*)base+128;memset(payload,0,bytes);*p=(T*)payload;halo_allocations[payload]={base,payload,bytes};return cudaSuccess;
}
static inline cudaError_t halo_free_host(void*p){if(!p)return cudaSuccess;halo_check_guards(p);auto a=halo_allocations.at(p);halo_allocations.erase(p);return cudaFreeHost(a.base);}
extern "C" void halo_set_callbacks(HaloCallback,HaloObserver);
extern "C" size_t halo_argument_size();
