// CPU-only address witness: original CUDA function bodies, synthetic index registers.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
using uint=unsigned int;
struct float3{float x,y,z;};
struct Dim{unsigned x=0;};Dim blockIdx,blockDim,threadIdx;
#define __device__
#define __global__
#define __forceinline__ inline
inline unsigned __umul24(unsigned a,unsigned b){return a*b;}
inline float __saturatef(float x){return std::max(0.f,std::min(1.f,x));}
bool observed_curve_case64=false;
#include "curve_instrumented_original.h"
int main(){const int w=16,h=16;std::vector<float3>image(w*h,float3{.75f,.75f,.75f});std::vector<uint>tmp(w*h),out(w*h);int path[][2]={{5,4},{4,5},{3,6},{2,6},{1,5},{1,4},{2,3},{3,2}};for(auto&pt:path)image[pt[0]*w+pt[1]]={.25f,.25f,.25f};blockDim.x=256;blockIdx.x=0;for(unsigned i=0;i<tmp.size();i++){threadIdx.x=i;check_connect(image.data(),tmp.data(),w,h);}threadIdx.x=4*w+4;eliminate_crosses(tmp.data(),out.data(),w,h);uint packed=rgbToyuv(float3{.25f,.25f,.25f});packed=(packed&0xffffff)<<8|0x10;printf("packed_island=%08x originalwholewordtest=%d graphlowbytetest=%d\n",packed,packed==0x10,(packed&255)==0x10);return observed_curve_case64?0:3;}
