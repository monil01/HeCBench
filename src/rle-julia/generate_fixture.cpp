#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <random>
#include <limits>
#include <algorithm>
using std::max;using std::min;bool g_verbose=false;
template<typename T>void DisplayResults(T*,int){}
#include "source_initialize.hpp"
int main(){std::ofstream f("fixture.bin",std::ios::binary);int32_t count=7;f.write((char*)&count,4);int seg[]={-1,1,16,256,4096,65536};for(int z=0;z<6;z++){int32_t n=1000000,kind=z==0?0:1,entropy=max(0,z-1);std::vector<int32_t>in(n);if(kind)Initialize(entropy,in.data(),n,seg[z]);else std::fill(in.begin(),in.end(),1);int32_t h[]={n,kind,entropy,seg[z]};f.write((char*)h,16);f.write((char*)in.data(),n*4);}
int32_t n=1031,kind=1,entropy=-1,maxseg=-1;std::vector<int32_t>a(n);int32_t keys[]={-2147483647-1,7,-1,2147483647,7,0};for(int i=0;i<n;i++)a[i]=keys[(i/17)%6];a[255]=-2147483647-1;a[256]=2147483647;a[511]=-1;a[512]=7;int32_t h[]={n,kind,entropy,maxseg};f.write((char*)h,16);f.write((char*)a.data(),n*4);}
