#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "../lulesh-cuda/lulesh.h"
inline void prepare_domain(Domain& d,int mode){
 int k=0;
 for(int r=0;r<d.numReg();r++){
  int rep=r<d.numReg()/2?1:r<(d.numReg()-(d.numReg()+15)/20)?1+d.cost():10*(1+d.cost());
  for(int e=0;e<d.regElemSize(r);e++){d.m_elemRep[d.regElemlist(r)[e]]=rep;d.m_elemElem[k++]=d.regElemlist(r)[e];}
 }
 if(mode){
  for(int i=0;i<d.numElem();i++){d.e(i)=0.;d.p(i)=0.;d.q(i)=0.;d.ss(i)=0.;}
  double alpha=mode==2?.01:mode==3?-.01:0.;
  for(int i=0;i<d.numNode();i++){d.xd(i)=alpha*d.x(i);d.yd(i)=alpha*d.y(i);d.zd(i)=alpha*d.z(i);}
  d.deltatime()=1e-4;
 }
}
inline std::vector<double> source_gamma(){
  Real_t  gamma[32] __attribute__((__aligned__(64)));
  gamma[0] = Real_t( 1.);
  gamma[1] = Real_t( 1.);
  gamma[2] = Real_t(-1.);
  gamma[3] = Real_t(-1.);
  gamma[4] = Real_t(-1.);
  gamma[5] = Real_t(-1.);
  gamma[6] = Real_t( 1.);
  gamma[7] = Real_t( 1.);
  gamma[8] = Real_t( 1.);
  gamma[9] = Real_t(-1.);
  gamma[10] = Real_t(-1.);
  gamma[11] = Real_t( 1.);
  gamma[12] = Real_t(-1.);
  gamma[13] = Real_t( 1.);
  gamma[14] = Real_t( 1.);
  gamma[15] = Real_t(-1.);
  gamma[16] = Real_t( 1.);
  gamma[17] = Real_t(-1.);
  gamma[18] = Real_t( 1.);
  gamma[19] = Real_t(-1.);
  gamma[20] = Real_t( 1.);
  gamma[21] = Real_t(-1.);
  gamma[22] = Real_t( 1.);
  gamma[23] = Real_t(-1.);
  gamma[24] = Real_t(-1.);
  gamma[25] = Real_t( 1.);
  gamma[26] = Real_t(-1.);
  gamma[27] = Real_t( 1.);
  gamma[28] = Real_t( 1.);
  gamma[29] = Real_t(-1.);
  gamma[30] = Real_t( 1.);
  gamma[31] = Real_t(-1.);
return std::vector<double>(gamma,gamma+32);}
