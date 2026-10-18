#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using uint=unsigned int;struct float3{float x,y,z;};struct Dim{unsigned x=0;};Dim blockIdx,blockDim,threadIdx;
#define __device__
#define __global__
#define __forceinline__ inline
inline unsigned __umul24(unsigned a,unsigned b){return a*b;}
inline float __saturatef(float x){return std::max(0.f,std::min(1.f,x));}
namespace Original{
#include "../../depixel-cuda/kernels.h"
}
namespace Corrected{
#include "../corrected_kernels.h"
}
int main(){static_assert(sizeof(float3)==12);FILE*f=fopen("/home/imo/HeCBench_julia/src/depixel-julia/fixture.bin","rb");std::vector<uint32_t>prefix(9998),gold(17200);if(!f||fread(prefix.data(),4,prefix.size(),f)!=prefix.size())return 4;fclose(f);f=fopen("/home/imo/HeCBench_julia/src/depixel-julia/oracle.bin","rb");if(!f||fread(gold.data(),4,gold.size(),f)!=gold.size())return 5;fclose(f);int offset=0,go=0;blockDim.x=256;for(int ci=0;ci<7;ci++){int w=prefix[9984+2*ci],h=prefix[9984+2*ci+1],n=w*h;std::vector<float3>img(n+16,float3{-12345.f,-12345.f,-12345.f});memcpy(img.data()+8,prefix.data()+offset,3*n*4);offset+=3*n;std::vector<uint>tmp(n+16,0xfedcba98),out(n+16,0xfedcba98);for(int i=0;i<n;i++){blockIdx.x=i/256;threadIdx.x=i%256;Corrected::check_connect(img.data()+8,tmp.data()+8,w,h);}for(int i=0;i<n;i++){blockIdx.x=i/256;threadIdx.x=i%256;Corrected::eliminate_crosses(tmp.data()+8,out.data()+8,w,h);}for(auto*arr:{&tmp,&out})for(uint x:*arr)if(x!=gold[go++])return 6;std::vector<uint>imagebits(3*n+48);memcpy(imagebits.data(),img.data(),imagebits.size()*4);for(uint x:imagebits)if(x!=gold[go++])return 7;
if(ci==0){std::vector<uint>ot(n),oo(n);for(int i=0;i<n;i++){blockIdx.x=i/256;threadIdx.x=i%256;Original::check_connect(img.data()+8,ot.data(),w,h);}for(int i=0;i<n;i++){blockIdx.x=i/256;threadIdx.x=i%256;Original::eliminate_crosses(ot.data(),oo.data(),w,h);}if(!std::equal(ot.begin(),ot.end(),tmp.begin()+8)||!std::equal(oo.begin(),oo.end(),out.begin()+8))return 8;printf("original-safe flat control CPU exactpass\n");}printf("corrected pipeline CPU fullgraphoracle exactpass case%d (%dx%d)\n",ci,w,h);}return go==17200?0:9;}
