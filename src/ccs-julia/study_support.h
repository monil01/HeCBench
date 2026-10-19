#pragma once
#include <bits/stdc++.h>
#ifndef CCS_ROOT
#define CCS_ROOT "."
#endif
struct Case{uint32_t n,D,m,repeat,ng,ne;float thr,overlap;std::vector<float>gene;std::vector<double>expected;};
inline std::vector<Case> cases(){std::ifstream f(std::string(CCS_ROOT)+"/fixtures.bin",std::ios::binary);uint32_t n;f.read((char*)&n,4);std::vector<Case>cs;for(uint32_t i=0;i<n;i++){Case c;f.read((char*)&c.n,24);f.read((char*)&c.thr,8);c.gene.resize(c.ng);c.expected.resize(c.ne);f.read((char*)c.gene.data(),c.ng*4);f.read((char*)c.expected.data(),c.ne*8);cs.push_back(c);}if(!f||f.peek()!=EOF)throw std::runtime_error("fixture");return cs;}
inline size_t outputzero(const Case&c){return 2*c.m*(c.D+c.n)+3*c.m;}
inline size_t prefix_count(const std::vector<Case>&cs){size_t n=0;for(auto&c:cs)n+=c.gene.size()+outputzero(c);return n;}
inline size_t output_count(const std::vector<Case>&cs){size_t n=prefix_count(cs);for(auto&c:cs)n+=c.expected.size();return n;}
inline void check(const std::vector<double>&v,const std::vector<Case>&cs){if(v.size()!=output_count(cs))throw std::runtime_error("artifactshape");size_t pos=0;auto eq=[&](double x){double y=v.at(pos++);if(y!=x)throw std::runtime_error("independentCCSoracle index="+std::to_string(pos-1)+" got="+std::to_string(y)+" expected="+std::to_string(x));};for(auto&c:cs){for(float x:c.gene)eq(x);for(size_t i=0;i<outputzero(c);i++)eq(0);}for(auto&c:cs)for(double x:c.expected)eq(x);}
