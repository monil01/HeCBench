#pragma once
#include <bits/stdc++.h>
#ifndef BMF_ROOT
#define BMF_ROOT "."
#endif
struct Case{uint32_t h,w,dim,iterations,weight,seed,runs,reduce_step,nc,ne;float temp,chance,reduce_factor,density,threshold;std::vector<uint32_t>C;std::vector<double>expected;};
inline std::vector<Case> cases(){std::ifstream f(std::string(BMF_ROOT)+"/fixtures.bin",std::ios::binary);uint32_t n;f.read((char*)&n,4);std::vector<Case>cs;for(uint32_t k=0;k<n;k++){Case c;f.read((char*)&c.h,40);f.read((char*)&c.temp,20);c.C.resize(c.nc);c.expected.resize(c.ne);f.read((char*)c.C.data(),c.nc*4);f.read((char*)c.expected.data(),c.ne*8);cs.push_back(c);}if(!f||f.peek()!=EOF)throw std::runtime_error("fixture");return cs;}
inline std::vector<uint32_t> padded(const Case&c){uint32_t pw=(c.w+31)/32*32;std::vector<uint32_t>v((c.h+31)/32*pw,0);for(uint32_t i=0;i<(c.h+31)/32;i++)for(uint32_t j=0;j<c.w;j++)v[i*pw+j]=c.C[i*c.w+j];return v;}
inline size_t prefix_count(const std::vector<Case>&cs){size_t n=0;for(auto&c:cs)n+=padded(c).size();return n;}
inline size_t output_count(const std::vector<Case>&cs){size_t n=prefix_count(cs);for(auto&c:cs)n+=c.ne+64;return n;}
inline void check(const std::vector<double>&v,const std::vector<Case>&cs){size_t p=0;if(v.size()!=output_count(cs))throw std::runtime_error("outputcount");for(auto&c:cs)for(uint32_t x:padded(c))if(v.at(p++)!=x)throw std::runtime_error("actualGPUinput");for(auto&c:cs){for(double x:c.expected){double y=v.at(p++);if(y!=x)throw std::runtime_error("fullscalaroracle index="+std::to_string(p-1)+" got="+std::to_string(y)+" expected="+std::to_string(x));}for(int i=0;i<64;i++)if(v.at(p++)!=3735928559.)throw std::runtime_error("guard");}}
