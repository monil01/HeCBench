
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define fp double
#define __device__
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/kernel_ecc.cu"
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/kernel_cam.cu"
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/kernel_fin.cu"
extern "C" void study_rhs64(double t,const double*y,const double*p,double*out,double*com){
 kernel_ecc((int)t,y,out,0,p);
 kernel_cam((int)t,y,out,46,p,0,com,0,y[35]*1e3);
 kernel_cam((int)t,y,out,61,p,5,com,1,y[36]*1e3);
 kernel_cam((int)t,y,out,76,p,10,com,2,y[37]*1e3);
}
extern "C" void study_fin64(double*y,double*p,double*out,double*com){kernel_fin(y,0,46,61,76,p,out,com[0],com[1],com[2]);}

#include <algorithm>
extern "C" void study_check(const double*v,size_t n,int xmax,const double*params,double*metrics){
 size_t head=3+(xmax+1)*92,calls=(size_t)v[2];double ma=0,mr=0;size_t bad=0,checked=0;double zero_bad=0;
 if(n!=head+278*calls||v[0]!=0||v[1]!=0){bad++;} // unmodified-source compensations are retained but not certified as finite source RHS.
 for(size_t k=0;k<calls;k++){
  auto t=v+head+278*k;double out[91]={},com[3]={};
  study_rhs64(t[0],t+2,params,out,com);
  for(int i=0;i<94;i++){double actual=t[93+i],expected=i<91?out[i]:com[i-91];double d=fabs(actual-expected);ma=std::max(ma,d);mr=std::max(mr,d/std::max(fabs(expected),1e-12));bad+=!std::isfinite(actual)||!std::isfinite(expected)||d>2e-4+2e-4*fabs(expected);checked++;}
  for(int i:{21,34,40,41})zero_bad+=t[93+i]!=0;
  // Full coupling, source host fin arithmetic Float64 crosscheck at actual state.
  double yi[91],pa[18];for(int i=0;i<91;i++)yi[i]=t[2+i];for(int i=0;i<18;i++)pa[i]=params[i];
  study_fin64(yi,pa,out,com);
  for(int i=0;i<91;i++){double d=fabs(t[187+i]-out[i]);ma=std::max(ma,d);bad+=!std::isfinite(t[187+i])||!std::isfinite(out[i])||d>2e-4+2e-4*fabs(out[i]);checked++;}
 }
 for(size_t i=0;i<n;i++)bad+=!std::isfinite(v[i]);
 metrics[0]=bad+zero_bad;metrics[1]=checked;metrics[2]=ma;metrics[3]=mr;metrics[4]=calls;metrics[5]=zero_bad;
}
