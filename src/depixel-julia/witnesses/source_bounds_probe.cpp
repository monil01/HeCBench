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
#include "../../depixel-cuda/kernels.h"
int main(){const int w=32,h=16;std::vector<float3>image(w*h);std::vector<uint>tmp(w*h),out(w*h);blockDim.x=256;for(int r=0;r<h;r++)for(int c=0;c<w;c++){float v=(r+c)%2?.75f:.25f;image[r*w+c]={v,v,v};}for(unsigned i=0;i<tmp.size();i++){blockIdx.x=i/256;threadIdx.x=i%256;check_connect(image.data(),tmp.data(),w,h);}int r=14,c=1;printf("source-produced checkerboard center[%d,%d] lowflags=%x neighbor=%x; rowlimit=%d, source_end_row=%d\n",r,c,tmp[r*w+c]&255,tmp[r*w+c+1]&255,h-1,(r<w-4)?r+4:w-1);fflush(stdout);blockIdx.x=(r*w+c)/256;threadIdx.x=(r*w+c)%256;eliminate_crosses(tmp.data(),out.data(),w,h);printf("out=%x\n",out[r*w+c]);}
