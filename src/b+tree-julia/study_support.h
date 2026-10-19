#pragma once
#include <vector>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#ifndef BT_ROOT
#define BT_ROOT "."
#endif
struct Case{int32_t n,nn,h,q;std::vector<int32_t>nodes,records,keys,start,end;};
inline std::vector<Case>cases(){std::ifstream f(std::string(BT_ROOT)+"/fixtures.bin",std::ios::binary);auto read=[&](auto&x){f.read((char*)&x,sizeof(x));if(!f)throw std::runtime_error("missing/shortfixture");};int32_t nc;read(nc);if(nc!=3)throw std::runtime_error("fixturecasecount");std::vector<Case>cs;for(int c=0;c<nc;c++){Case x;read(x.n);read(x.nn);read(x.h);read(x.q);if(x.n<1||x.nn<1||x.h<0||x.h>2||x.q<1||x.n>65536)throw std::runtime_error("fixtureshape");auto vread=[&](std::vector<int32_t>&v,int n){v.resize(n);f.read((char*)v.data(),n*4);if(!f)throw std::runtime_error("shortfixturearray");};vread(x.nodes,x.nn*517);vread(x.records,x.n);vread(x.keys,x.q);vread(x.start,x.q);vread(x.end,x.q);cs.push_back(x);}if(f.peek()!=EOF)throw std::runtime_error("fixturetrailing");return cs;}
inline int32_t nodekey(const Case&c,int id,int lane){return c.nodes.at(id*517+258+lane);}
inline int32_t nodeindex(const Case&c,int id,int lane){return c.nodes.at(id*517+1+lane);}
inline int leaf(const Case&c,int32_t key){int id=0;for(int level=0;level<c.h;level++){int found=-1;for(int lane=0;lane<256;lane++)if(nodekey(c,id,lane)<=key&&key<nodekey(c,id,lane+1)){if(found!=-1)throw std::runtime_error("ambiguousnodeinterval");found=lane;}if(found<0)throw std::runtime_error("nointerval");id=nodeindex(c,id,found);if(id<0||id>=c.nn)throw std::runtime_error("childrange");}return id;}
inline void topology(const Case&c){std::vector<int>seen(c.nn,0),records(c.n,0);struct Work{int id,level;int64_t lo,hi;};std::vector<Work>work{{0,0,INT32_MIN,INT32_MAX}};while(!work.empty()){auto w=work.back();work.pop_back();if(w.id<0||w.id>=c.nn||seen[w.id]++)throw std::runtime_error("topologycycle");int base=w.id*517,nk=c.nodes[base+516];bool isleaf=c.nodes[base+515]!=0;if(nk<2||nk>257||isleaf!=(w.level==c.h)||nodekey(c,w.id,0)!=INT32_MIN||nodekey(c,w.id,nk-1)!=INT32_MAX)throw std::runtime_error("topologyheader");if(isleaf){for(int j=1;j<nk-1;j++){int key=nodekey(c,w.id,j),record=nodeindex(c,w.id,j);if(record<0||record>=c.n||records[record]++||key!=3*record-2*c.n||c.records[record]!=key*7+3||key<w.lo||key>=w.hi)throw std::runtime_error("leafrankpartition");}}else for(int j=0;j<nk-1;j++){int64_t lo=std::max(w.lo,int64_t(nodekey(c,w.id,j))),hi=std::min(w.hi,int64_t(nodekey(c,w.id,j+1)));if(lo>=hi)throw std::runtime_error("emptypartition");work.push_back({nodeindex(c,w.id,j),w.level+1,lo,hi});}}for(int n:seen)if(n!=1)throw std::runtime_error("unreachableknode");for(int n:records)if(n!=1)throw std::runtime_error("incompleterecords");}
inline long prefixcount(const std::vector<Case>&cs){long n=1;for(auto&c:cs)n+=4+c.nodes.size()+c.records.size()+12*c.q;return n;}
inline long outcount(const std::vector<Case>&cs){long n=prefixcount(cs);for(auto&c:cs)n+=9*c.q+1792;return n;}
extern "C" int study_check(const char*,double*);
