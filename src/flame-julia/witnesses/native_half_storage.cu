// CPU-only witness of the exact typed conversions in the original GPU kernels.
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
int main(){static_assert(sizeof(__half)==sizeof(short),"16bitstorage");for(float x:{-1.25f,-.75f,.375f,.75f,1.25f}){__half h=__float2half_rn(x);short stored=h;float decoded=__half2float(stored);unsigned short bits;memcpy(&bits,&h,2);printf("input %.9g source_short %d source_decode %.9g intended_half_bits %04x half_decode %.9g\n",x,int(stored),decoded,bits,__half2float(h));}return 0;}
