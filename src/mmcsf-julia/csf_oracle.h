#pragma once
#include <array>
#include <cstring>
#include <stdexcept>
struct MMCOO{std::array<uint32_t,4>i{};float v;};
struct MMEntry{uint64_t key;uint32_t bits;bool operator<(const MMEntry&b)const{return key<b.key||(key==b.key&&bits<b.bits);}bool operator==(const MMEntry&b)const{return key==b.key&&bits==b.bits;}};
inline MMEntry mm_entry(const std::array<uint32_t,4>&c,float v,const std::vector<size_t>&dims){uint64_t key=0;for(size_t j=0;j<dims.size();j++){if(c[j]>=dims[j])throw std::runtime_error("COOcoordinateoutofbounds");key=key*dims[j]+c[j];}uint32_t bits;memcpy(&bits,&v,4);return{key,bits};}
inline void mm_check_csf(const TiledTensor*parts,int nd,const std::vector<size_t>&dims,const std::vector<MMCOO>&coo,bool exact_multiset){
 std::vector<MMEntry>a,b;if(exact_multiset){a.reserve(coo.size());b.reserve(coo.size());for(auto&x:coo)a.push_back(mm_entry(x.i,x.v,dims));}
 size_t total=0;
 for(int p=0;p<nd;p++){
  const auto&t=parts[p];if(!t.totNnz)continue;size_t seen=0;
  if(t.fbrPtr[0].size()!=t.fbrIdx[0].size()+1)throw std::runtime_error("CSFslicePointershape");
  for(size_t sl=0;sl<t.fbrIdx[0].size();sl++)for(size_t f=t.fbrPtr[0].at(sl);f<t.fbrPtr[0].at(sl+1);f++){
   auto visit=[&](size_t z,size_t in){
    if(z>=t.totNnz)throw std::runtime_error("CSFnonzeroaddressbound");
    std::array<uint32_t,4>ix{};ix[t.modeOrder[0]]=t.fbrIdx[0].at(sl);ix[t.modeOrder[1]]=t.fbrIdx[1].at(f);
    if(nd==4){ix[t.modeOrder[2]]=t.fbrIdx[2].at(in);ix[t.modeOrder[3]]=t.inds[t.modeOrder[3]][z];}
    else ix[t.modeOrder[2]]=t.inds[t.modeOrder[2]][z];
    for(int j=0;j<nd;j++)if(ix[j]!=t.inds[j][z]||ix[j]>=dims[j])throw std::runtime_error("CSFcoordinatevsCOOmismatch");
    if(t.fbrLikeSlcInds[f]!=ix[t.modeOrder[0]])throw std::runtime_error("CSFsliceMapMismatch");
    if(exact_multiset)b.push_back(mm_entry(ix,t.vals[z],dims));seen++;
   };
   if(nd==3){for(size_t z=t.fbrPtr[1].at(f);z<t.fbrPtr[1].at(f+1);z++)visit(z,0);}
   else for(size_t in=t.fbrPtr[1].at(f);in<t.fbrPtr[1].at(f+1);in++)for(size_t z=t.fbrPtr[2].at(in);z<t.fbrPtr[2].at(in+1);z++)visit(z,in);
  }
  if(seen!=t.totNnz)throw std::runtime_error("CSFcoveragecount");total+=seen;
 }
 if(total!=coo.size())throw std::runtime_error("Partitioncoveragecount");
 if(exact_multiset){std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());if(a!=b)throw std::runtime_error("CSFvsoriginalCOOexactmultisetmismatch");}
}
