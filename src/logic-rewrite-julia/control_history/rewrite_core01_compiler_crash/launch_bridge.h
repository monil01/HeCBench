#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <stdexcept>
#include <cuda_runtime.h>
struct LRLaunch {const char *name;uint64_t geometry[6];uint64_t nargs;uint64_t kinds[32];uint64_t values[32];};
using LRCallback=int(*)(const LRLaunch*);
extern LRCallback lr_callback;
template<class T>void lr_arg(LRLaunch &d,int i,T v){
 static_assert(std::is_pointer_v<T>||std::is_arithmetic_v<T>,"Only scalar/pointer kernelarguments supported");
 if constexpr(std::is_pointer_v<T>){d.kinds[i]=0;d.values[i]=reinterpret_cast<uintptr_t>(v);}
 else if constexpr(std::is_same_v<T,bool>){d.kinds[i]=3;d.values[i]=v;}
 else if constexpr(std::is_floating_point_v<T>){d.kinds[i]=sizeof(T)==4?4:5;memcpy(&d.values[i],&v,sizeof(v));}
 else if constexpr(std::is_signed_v<T>){d.kinds[i]=1;d.values[i]=uint64_t(int64_t(v));}
 else{d.kinds[i]=2;d.values[i]=uint64_t(v);}
}
template<class F,class...A>void lr_launch(const char*name,dim3 grid,dim3 block,F native,A...args){
 if(!lr_callback){native();return;}
 LRLaunch d{};d.name=name;d.geometry[0]=grid.x;d.geometry[1]=grid.y;d.geometry[2]=grid.z;d.geometry[3]=block.x;d.geometry[4]=block.y;d.geometry[5]=block.z;d.nargs=sizeof...(A);static_assert(sizeof...(A)<=32);int i=0;(lr_arg(d,i++,args),...);
 if(lr_callback(&d))throw std::runtime_error("Julia device callback rejected source kernel; native fallback forbidden");
}
