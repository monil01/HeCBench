#include <hip/hip_runtime.h>
#include <cstdint>
#define WARP_MASK 0xffffffff
#define WARP_WIDTH 32
#define LANEID_REVERSED(lane) (31-(lane))
__device__ unsigned lane_id(){return threadIdx.x & 31;}
#define __shfl_sync(mask,v,l,...) __shfl(v,l,32)
#define __shfl_up_sync(mask,v,l,...) __shfl_up(v,l,32)
#define __shfl_down_sync(mask,v,l,...) __shfl_down(v,l,32)
#define __ballot_sync(mask,p) ((uint32_t)(__ballot(p) >> (threadIdx.x & 32)))
struct PoolAllocator {
 uint32_t *pool,*count;
 __device__ uint32_t allocate(){return atomicAdd(count,1u);}
 __device__ uint32_t* getAddressPtr(uint32_t address){return pool+size_t(address)*32;}
};
#include "map_warps.hpp"
extern "C" __global__ void clear_pool(uint32_t *pool, size_t n){
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n) pool[i]=0;
}
extern "C" __global__ void init_tree(uint32_t *pool,uint32_t *count,uint32_t *root){
 if(threadIdx.x==0){*count=1;*root=0;pool[0]=1;pool[1]=0;}
}
extern "C" __global__ void insert_keys(uint32_t *root,uint32_t *keys,uint32_t *values,uint32_t n,uint32_t *pool,uint32_t *count){
 uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x,lane=lane_id();
 if(tid-lane>=n)return;
 uint32_t key=0,value=0;
 bool work=tid<n;
 if(work){key=keys[tid]+2;value=values[tid]+2;}
 PoolAllocator a{pool,count};
 GpuBTree::warps::insertion_unit(work,key,value,root,&a);
}
extern "C" __global__ void search_keys(uint32_t *root,uint32_t *queries,uint32_t *results,uint32_t n,uint32_t *pool,uint32_t *count){
 uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x,lane=lane_id();
 if(tid-lane>=n)return;
 uint32_t key=tid<n?queries[tid]+2:0,result=0;
 bool work=tid<n;
 PoolAllocator a{pool,count};
 GpuBTree::warps::search_unit(work,lane,key,result,root,&a);
 if(tid<n)results[tid]=result?result-2:0;
}
