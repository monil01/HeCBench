#pragma once
#include <array>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cfloat>
using NBBytes=std::vector<std::vector<char>>;
template<class T>static T nb_get(const std::vector<char>&b,size_t k){if((k+1)*sizeof(T)>b.size())throw std::runtime_error("Oracleinputindexbound");T v;memcpy(&v,b.data()+k*sizeof(T),sizeof(T));return v;}
struct NBOracle{std::vector<std::array<long double,3>>force,shift;size_t pairs=0,excludedPairs=0,cutoffPairs=0,selfRejected=0;};
static long double nb_corr(long double x){long double pi=acosl(-1.L);if(x<.01L){long double sum=0,t=1;for(int k=0;k<40;k++){sum+=t/(2*k+3);t*=-x/(k+1);}return-4/sqrtl(pi)*sum;}long double z=sqrtl(x);return 2*expl(-x)/(sqrtl(pi)*x)-erfl(z)/(x*z);}
static NBOracle nb_oracle(const NBLaunch&q,const NBBytes&b){
 NBOracle g;g.force.resize(q.nAtoms);g.shift.resize(45);for(int i=0;i<q.nAtoms;i++){auto f=nb_get<Float3>(b[1],i);for(int k=0;k<3;k++)g.force[i][k]=f[k];}for(int i=0;i<45;i++){auto f=nb_get<Float3>(b[3],i);for(int k=0;k<3;k++)g.shift[i][k]=f[k];}
 for(int block=0;block<q.gridZ;block++){
  auto sci=nb_get<nbnxn_sci_t>(b[5],block);if(sci.sci<0||sci.shift<0||sci.shift>=45||sci.cj4_ind_start<0||sci.cj4_ind_end>q.nCJ)throw std::runtime_error("OracleSCIbound");auto sh=nb_get<Float3>(b[2],sci.shift);
  for(int j4=sci.cj4_ind_start;j4<sci.cj4_ind_end;j4++){
   auto cj4=nb_get<nbnxn_cj4_t>(b[4],j4);
   for(int j=0;j<8;j++){
    int warp=j/4;uint32_t mask=cj4.imei[warp].imask;if(!mask)continue;auto ex=nb_get<nbnxn_excl_t>(b[6],cj4.imei[warp].excl_ind);
    for(int jm=0;jm<4;jm++)for(int ciOffset=0;ciOffset<8;ciOffset++){
     uint32_t bit=uint32_t(1)<<(jm*8+ciOffset);if(!(mask&bit))continue;int ci=sci.sci*8+ciOffset,cj=cj4.cj[jm],aj=cj*8+j;
     for(int i=0;i<8;i++){
      int ai=ci*8+i;if(ai<0||ai>=q.nAtoms||aj<0||aj>=q.nAtoms)throw std::runtime_error("Oracleatombound");if(sci.shift==c_centralShiftIndex&&j<=i&&ci==cj){g.selfRejected++;continue;}
      auto xi=nb_get<Float4>(b[0],ai),xj=nb_get<Float4>(b[0],aj);std::array<long double,3>r={(long double)xi.x+sh[0]-xj.x,(long double)xi.y+sh[1]-xj.y,(long double)xi.z+sh[2]-xj.z};long double r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2];if(!(r2<q.cutoffSq)){g.cutoffPairs++;continue;}
      r2=std::max(r2,(long double)c_nbnxnMinDistanceSquared);long double inv=1/sqrtl(r2),inv2=1/r2,inv6=inv2*inv2*inv2;bool included=(ex.pair[(j*8+i)&31]&bit)!=0;int ti=nb_get<int>(b[7],ai),tj=nb_get<int>(b[7],aj);if(ti<0||ti>=q.numTypes||tj<0||tj>=q.numTypes)throw std::runtime_error("Oracletypebound");auto coeff=nb_get<Float2>(b[8],q.numTypes*ti+tj);long double lj=included?inv6*((long double)coeff.y*inv6-coeff.x)*inv2:0;long double beta=q.beta,qq=(long double)q.epsFac*xi.w*xj.w;long double elec=qq*((included?inv2*inv:0)+beta*beta*beta*nb_corr(beta*beta*r2));long double scale=lj+elec;
      for(int k=0;k<3;k++){long double f=r[k]*scale;g.force[ai][k]+=f;g.force[aj][k]-=f;if(q.calcShift&&sci.shift!=c_centralShiftIndex)g.shift[sci.shift][k]+=f;}
      g.pairs++;g.excludedPairs+=!included;
     }
    }
   }
  }
 }
 return g;
}
