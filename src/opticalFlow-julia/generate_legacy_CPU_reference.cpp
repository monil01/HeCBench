// Original source CPU checker, outside every GPU ROI. Defined unused padding only.
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <new>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include "original_native/flowGold.h"
void*operator new[](std::size_t n){void*p=std::malloc(n?n:1);if(!p)throw std::bad_alloc();std::memset(p,0,n);return p;}
void operator delete[](void*p)noexcept{std::free(p);}
void operator delete[](void*p,std::size_t)noexcept{std::free(p);}
int main(int argc,char**argv){if(argc!=3)return 2;std::ifstream f(argv[1],std::ios::binary);uint32_t h[8];float alpha;f.read((char*)h,32);f.read((char*)&alpha,4);if(!f||h[0]!=0x4f463537)return 3;int w=h[1],height=h[2],s=h[3];std::vector<float>a(s*height),b(s*height),u(s*height),v(s*height);f.read((char*)a.data(),a.size()*4);f.read((char*)b.data(),b.size()*4);if(!f||f.peek()!=EOF)return 4;ComputeFlowGold(a.data(),b.data(),w,height,s,alpha,h[4],h[5],h[6],u.data(),v.data());std::ofstream o(argv[2],std::ios::binary);o.write((char*)&h[1],8);for(auto*p:{&u,&v})for(int y=0;y<height;y++)o.write((char*)(p->data()+y*s),w*4);return o?0:5;}
