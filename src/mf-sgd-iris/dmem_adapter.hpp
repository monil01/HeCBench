#pragma once
#include <cstddef>
#include <hip/hip_fp16.h>
struct mf_node;
using dmemError_t=int;
using dmemStream_t=int;
constexpr int dmemSuccess=0;
enum dmemMemcpyKind { dmemMemcpyHostToDevice, dmemMemcpyDeviceToHost };
void dmem_platform_init(int*,char***);
void dmem_platform_finalize();
int dmemGetDeviceCount(int*);
int dmemSetDevice(int);
int dmemMalloc(void**,std::size_t);
int dmemMallocHost(void**,std::size_t);
int dmemFree(void*);
int dmemFreeHost(void*);
int dmemMemcpy(void*,const void*,std::size_t,dmemMemcpyKind);
int dmemMemcpyAsync(void*,const void*,std::size_t,dmemMemcpyKind,dmemStream_t);
int dmemDeviceSynchronize();
int dmemPeekAtLastError();
int dmemStreamCreate(dmemStream_t*);
const char*dmemGetErrorString(int);
void launch_init_rand_state(std::size_t,std::size_t,unsigned,unsigned*);
void launch_init_rand_state(std::size_t,std::size_t,unsigned,unsigned*,int);
void launch_random_init(std::size_t,std::size_t,unsigned*,int,half*,long long,long long,float);
void launch_transform_half(std::size_t,std::size_t,const half*,float*,long long);
void launch_sgd_k128_kernel_hogwild_warp32_lrate(std::size_t,std::size_t,const mf_node*,long long,half*,half*,unsigned*,const float*,long long,long long,int,int,int,int,int,int,float,float,int,int,int,int);
