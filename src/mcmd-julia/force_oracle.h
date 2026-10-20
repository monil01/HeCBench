#pragma once
// Independent all-pairs long-double real-space forces and factorized reciprocal
// structure factors. No native GPU/CPU force function is used by this oracle.
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
struct MAtom{double pos[3],eps,sig,charge,f[3];int molid,frozen;double u[3],polar;};
using LV=std::array<long double,3>;
static std::vector<LV> independent_forces(const MAtom*a,int N,int pf,long double cut,long double alpha,int km,int ks,const double*b,const double*rb){
 if(pf!=1)throw std::runtime_error("Named full source fixture is LJ+Ewald; other potential branches not certified");
 std::vector<LV> f(N,LV{0,0,0});const long double pi=acosl(-1.0L);
 for(int i=0;i<N;i++)for(int j=i+1;j<N;j++){
  if(a[i].molid==a[j].molid||(a[i].frozen&&a[j].frozen))continue;
  LV d,img,x;for(int c=0;c<3;c++)d[c]=(long double)a[i].pos[c]-a[j].pos[c];
  for(int c=0;c<3;c++){img[c]=0;for(int q=0;q<3;q++)img[c]+=(long double)rb[c*3+q]*d[q];img[c]=nearbyintl(img[c]);}
  long double r2=0;for(int c=0;c<3;c++){x[c]=d[c];for(int q=0;q<3;q++)x[c]-=(long double)b[c*3+q]*img[q];r2+=x[c]*x[c];}
  if(r2>cut*cut)continue;if(!(r2>0))throw std::runtime_error("Coincident atoms outside declared domain");
  long double r=sqrtl(r2),sig=((long double)a[i].sig+a[j].sig)/2,eps=sqrtl((long double)a[i].eps*a[j].eps),s6=powl(sig,6),r6=r2*r2*r2;
  long double radial=24*eps*(2*s6*s6/(r6*r6*r2)-s6/(r6*r2));
  long double qq=(long double)a[i].charge*a[j].charge;
  radial+=qq*(2*alpha*expl(-alpha*alpha*r2)/(sqrtl(pi)*r2)+erfcl(alpha*r)/(r2*r));
  for(int c=0;c<3;c++){long double v=radial*x[c];f[i][c]+=v;f[j][c]-=v;}
 }
 if(ks){
  long double det=(long double)b[0]*(b[4]*b[8]-b[5]*b[7])-(long double)b[1]*(b[3]*b[8]-b[5]*b[6])+(long double)b[2]*(b[3]*b[7]-b[4]*b[6]);
  int nm=0;for(int i=0;i<N;i++)nm=std::max(nm,a[i].molid+1);
  std::vector<long double> si(N),co(N),ms(nm),mc(nm),fs(nm),fc(nm);
  for(int l0=0;l0<=km;l0++)for(int l1=l0?-km:0;l1<=km;l1++)for(int l2=(!l0&&!l1)?1:-km;l2<=km;l2++){
   if(l0*l0+l1*l1+l2*l2>km*km)continue;int l[3]={l0,l1,l2};LV k{0,0,0};for(int c=0;c<3;c++)for(int q=0;q<3;q++)k[c]+=2*pi*(long double)rb[3*q+c]*l[q];long double k2=k[0]*k[0]+k[1]*k[1]+k[2]*k[2];
   std::fill(ms.begin(),ms.end(),0);std::fill(mc.begin(),mc.end(),0);std::fill(fs.begin(),fs.end(),0);std::fill(fc.begin(),fc.end(),0);long double ts=0,tc=0,tf=0,cf=0;
   for(int i=0;i<N;i++){long double phase=0;for(int c=0;c<3;c++)phase+=k[c]*a[i].pos[c];si[i]=sinl(phase);co[i]=cosl(phase);long double qs=a[i].charge*si[i],qc=a[i].charge*co[i];ts+=qs;tc+=qc;ms[a[i].molid]+=qs;mc[a[i].molid]+=qc;if(a[i].frozen){tf+=qs;cf+=qc;fs[a[i].molid]+=qs;fc[a[i].molid]+=qc;}}
   long double pref=8*pi/det*expl(-k2/(4*alpha*alpha))/k2;
   for(int i=0;i<N;i++){long double S=ts-ms[a[i].molid],C=tc-mc[a[i].molid];if(a[i].frozen){S-=tf-fs[a[i].molid];C-=cf-fc[a[i].molid];}long double v=pref*a[i].charge*(si[i]*C-co[i]*S);for(int c=0;c<3;c++)f[i][c]+=v*k[c];}
  }
 }
 return f;
}
