#pragma once
#include <bits/stdc++.h>
#include <zlib.h>
#ifndef DERED_ROOT
#define DERED_ROOT "."
#endif
struct Case{uint32_t n,k,total,np,ne,nz;float threshold;std::vector<int>lengths;std::vector<long>offsets;std::vector<uint8_t>raw;std::vector<int64_t>expected;};
inline std::vector<Case> cases(){static_assert(sizeof(long)==8);std::ifstream f(std::string(DERED_ROOT)+"/fixtures.bin",std::ios::binary);uint32_t n;f.read((char*)&n,4);std::vector<Case>cs;for(uint32_t i=0;i<n;i++){Case c;f.read((char*)&c.n,24);f.read((char*)&c.threshold,4);c.lengths.resize(c.n);c.offsets.resize(c.n+1);c.raw.resize(c.total);c.expected.resize(c.ne);f.read((char*)c.lengths.data(),c.n*4);f.read((char*)c.offsets.data(),(c.n+1)*8);f.read((char*)c.raw.data(),c.total);std::vector<uint8_t>z(c.nz);f.read((char*)z.data(),z.size());uLongf size=c.ne*8;if(uncompress((Bytef*)c.expected.data(),&size,z.data(),z.size())!=Z_OK||size!=c.ne*8)throw std::runtime_error("oraclezlib");cs.push_back(c);}if(!f||f.peek()!=EOF)throw std::runtime_error("fixture");return cs;}
inline size_t prefix_count(const std::vector<Case>&cs){size_t n=0;for(auto&c:cs)n+=10*c.n+2+3*c.total+c.np+65536;return n;}
inline size_t output_count(const std::vector<Case>&cs){size_t n=prefix_count(cs);for(auto&c:cs)n+=c.ne;return n;}
inline void check(const std::vector<int64_t>&v,const std::vector<Case>&cs){if(v.size()!=output_count(cs))throw std::runtime_error("artifactcount");size_t p=0;auto eq=[&](int64_t e){if(v.at(p++)!=e)throw std::runtime_error("independentintegerpipeline index="+std::to_string(p-1)+" got="+std::to_string(v[p-1])+" expected="+std::to_string(e));};for(auto&c:cs){for(int x:c.lengths)eq(x);for(long x:c.offsets)eq(x);for(uint8_t x:c.raw)eq(x);for(size_t i=0;i<c.np+7*c.n+2*c.total;i++)eq(0);for(uint32_t i=0;i<c.n;i++)eq(-1);for(int i=0;i<65536;i++)eq(0);eq(-1);}for(auto&c:cs)for(int64_t e:c.expected)eq(e);}
