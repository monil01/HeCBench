#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
using I=int32_t;
inline void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
inline std::vector<I>readints(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);need(f.good(),"missing binary");auto n=f.tellg();need(n>=0&&n%4==0,"binary size");f.seekg(0);std::vector<I>v(size_t(n)/4);f.read((char*)v.data(),n);need(f.good(),"short read");return v;}
struct Case{int nets,pins;std::string name;std::vector<I>idx,x,y,oracle;};
inline std::vector<Case>cases(){std::vector<Case>cs;for(const char*name:{"original_first8192_plus_degree_coverage","degree_tail_duplicate_collinear_controls"}){Case c;c.name=name;auto v=readints("data/"+c.name+".bin");need(v.size()>2,"fixture");c.nets=v[0];c.pins=v[1];need(v.size()==size_t(3+c.nets+2*c.pins),"fixture length");auto a=v.begin()+2;c.idx.assign(a,a+c.nets+1);a+=c.nets+1;c.x.assign(a,a+c.pins);a+=c.pins;c.y.assign(a,a+c.pins);c.oracle=readints("data/"+c.name+"_oracle.bin");need(c.oracle.size()>=size_t(c.nets+2+8*c.pins),"oracle length");cs.push_back(std::move(c));}return cs;}
extern "C" int study_check(const char*,double*);
