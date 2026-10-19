#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <zlib.h>
#ifndef FACE_ROOT
#define FACE_ROOT "."
#endif
struct FCase{int w,h,mode;std::vector<int32_t>input;std::vector<int64_t>expected;};
inline std::vector<FCase>face_load(){std::ifstream f(std::string(FACE_ROOT)+"/fixtures.bin",std::ios::binary);if(!f)throw std::runtime_error("missing face fixture");auto rd=[&](){uint32_t x;f.read((char*)&x,4);if(!f)throw std::runtime_error("truncated face fixture");return x;};auto n=rd();std::vector<FCase>cs;for(unsigned k=0;k<n;k++){FCase c;c.w=rd();c.h=rd();c.mode=rd();auto ni=rd(),ne=rd(),nz=rd();c.input.resize(ni);c.expected.resize(ne);std::vector<unsigned char>z(nz);f.read((char*)c.input.data(),ni*4);f.read((char*)z.data(),nz);uLongf nb=ne*8;if(uncompress((Bytef*)c.expected.data(),&nb,z.data(),nz)!=Z_OK||nb!=ne*8||!f)throw std::runtime_error("bad oracle blob");cs.push_back(c);}return cs;}
inline std::vector<int64_t>face_expected(){std::vector<int64_t>v;auto cs=face_load();for(auto&c:cs){v.insert(v.end(),c.input.begin(),c.input.end());auto r=c.input.begin()+c.w*c.h+25;v.insert(v.end(),r,r+2913*12);}for(auto&c:cs)v.insert(v.end(),c.expected.begin(),c.expected.end());v.insert(v.end(),768,165);return v;}
inline bool face_check(const int64_t*x,size_t n){auto e=face_expected();if(n!=e.size()){std::cerr<<"shape "<<n<<" != "<<e.size()<<"\n";return false;}int wrong=0;for(size_t i=0;i<n;i++)if(e[i]!=x[i]){if(wrong<8)std::cerr<<"wrong "<<i<<" "<<x[i]<<" != "<<e[i]<<"\n";wrong++;}if(wrong)std::cerr<<"total wrong="<<wrong<<"\n";return !wrong;}
