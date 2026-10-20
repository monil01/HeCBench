#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
struct Tensor;struct TiledTensor;struct Matrix;class Options;
struct MMLaunch{
 int32_t op,grid,block,ndims;
 uint64_t nfibers,rank,wps;
 int32_t logWPC,fpw,logFPW,padding;
 uint64_t mode;
 const void* pointers[12];uint64_t lengths[12];
};
static_assert(sizeof(MMLaunch)==256,"MMLaunch ABI");
using MMCallback=int(*)(const MMLaunch*);
extern MMCallback mm_callback;
cudaError_t mm_malloc(void**,size_t);
cudaError_t mm_free(void*);
void mm_begin_case(const Tensor&,Matrix*,const Options&);
void mm_mode_begin(int,float*,const unsigned int*,const TiledTensor*);
void mm_mode_end(int,float*,const unsigned int*,const TiledTensor*);
void mm_dispatch(const MMLaunch&);
void mm_native_launch(const MMLaunch&);
MMLaunch mm_make3(int,uint32_t,uint32_t,const float*,const size_t*,const size_t*,const size_t*,const size_t*,const size_t*,size_t,float*,const float*,const float*,size_t,size_t,size_t,int,int=1,int=0);
MMLaunch mm_make4(int,uint32_t,uint32_t,const float*,const size_t*,const size_t*,const size_t*,const size_t*,const size_t*,const size_t*,const size_t*,size_t,float*,const float*,const float*,const float*,size_t,size_t,size_t,int,int=1,int=0);
