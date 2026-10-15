#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include "format.h"
__device__ unsigned code_at(const uint8_t*p,size_t i){size_t bit=i*FORMAT_BITS,byte=bit/8;unsigned code=p[byte]>>(bit%8);if(bit%8+FORMAT_BITS>8)code|=unsigned(p[byte+1])<<(8-bit%8);return code&((1u<<FORMAT_BITS)-1);}
__device__ float value(unsigned code){unsigned sign=code>>(FORMAT_BITS-1),mant=code&((1<<MANTISSA_BITS)-1),exp=(code>>MANTISSA_BITS)&((1<<(FORMAT_BITS-MANTISSA_BITS-1))-1);float v=exp?ldexpf(1+float(mant)/(1<<MANTISSA_BITS),int(exp)-EXPONENT_BIAS):ldexpf(float(mant)/(1<<MANTISSA_BITS),1-EXPONENT_BIAS+SUBNORMAL_SHIFT);return sign?-v:v;}
extern "C" __global__ void decode(const uint8_t*packed,const uint8_t*scales,float*out,int rows,int K){
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size_t(rows)*K)return;
 uint8_t s=scales[(i/K)*(K/SCALE_BLOCK)+(i%K)/SCALE_BLOCK];
 float scale=SCALE_E8M0?ldexpf(1,int(s)-127):((s>>3)?ldexpf(1+float(s&7)/8,int(s>>3)-7):ldexpf(float(s&7)/8,-6));
 out[i]=value(code_at(packed,i))*scale;
}
extern "C" __global__ void bf16(const float*in,uint16_t*out,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){union{float f;uint32_t u;}v;v.f=in[i];v.u+=0x7fff+((v.u>>16)&1);out[i]=v.u>>16;}}
