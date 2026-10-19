// Independent generic SHA-256 message-schedule recurrence, no CUDA source/macros.
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <vector>
static uint32_t rotr(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
int main(int argc,char**argv){assert(argc==2);FILE*f=fopen(argv[1],"wb");assert(f);std::vector<uint32_t>chunk;chunk.reserve(64*4096);for(uint64_t count=0;count<1048576;count++){
 uint8_t bytes[64]={};for(int i=0;i<16;i++)bytes[i]=i;for(int i=0;i<8;i++)bytes[16+i]=count>>(8*i);bytes[24]=0x80;bytes[62]=2;bytes[63]=0xc0;
 uint32_t w[64];for(int j=0;j<16;j++)w[j]=(uint32_t(bytes[j*4])<<24)|(uint32_t(bytes[j*4+1])<<16)|(uint32_t(bytes[j*4+2])<<8)|bytes[j*4+3];
 for(int j=16;j<64;j++){uint32_t x=w[j-15],y=w[j-2];w[j]=w[j-16]+(rotr(x,7)^rotr(x,18)^(x>>3))+w[j-7]+(rotr(y,17)^rotr(y,19)^(y>>10));}
 chunk.insert(chunk.end(),w,w+64);if(chunk.size()==64*4096){assert(fwrite(chunk.data(),4,chunk.size(),f)==chunk.size());chunk.clear();}
 }assert(chunk.empty());assert(fclose(f)==0);return 0;}
