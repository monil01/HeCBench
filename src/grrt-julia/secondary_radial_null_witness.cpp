#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include "../grrt-cuda/constants.h"
namespace original {
#include "study_cpu_clone_corrected.hpp"
}
namespace canonical {
#include "secondary_schwarzschild_canonical_clone.hpp"
}
template<class D,class R>void run(const char*name,D derivative,R rk){
 double input[4]={0,45,8,340e9},V[9]={1000,PI/4,0,2.00001,6,0,0,0,0};double state[6]={1000,PI/4,0,0,1000./998.,0},d[6],scale[6];double h=.5,hdid=0;int steps=0,next=0;double thresholds[]={500,100,20,6,3,2.1,2.01,2.00001};double maxv=0;long double maxH=0;printf("{\"method\":\"%s\",\"snapshots\":[",name);
 while(state[0]>2.00001 && steps<2000000){derivative(V,input,state,d);for(int i=0;i<6;i++)scale[i]=fabs(state[i])+fabs(d[i]*h)+1e-3;h=rk(V,input,state,d,h,1e14,scale,&hdid);steps++;derivative(V,input,state,d);maxv=std::max(maxv,fabs(d[0]+1));long double r=state[0],pr=state[4],f=(r-2)/r,H=-1/f+f*pr*pr;maxH=std::max(maxH,fabsl(H));while(next<8 && state[0]<=thresholds[next]){printf("%s{\"threshold\":%.17g,\"r\":%.17g,\"radial_velocity\":%.17g,\"H_Float80_from_Float64state\":%.21Lg,\"H_r3_rminus2\":%.21Lg}",next?",":"",thresholds[next],state[0],d[0],H,H*r*r*r*(r-2));next++;}}
 printf("],\"accepted_steps\":%d,\"max_abs_radial_velocity_error_vs_analytic_null_minus1\":%.17g,\"max_abs_H_evaluated_Float80\":%.21Lg}\n",steps,maxv,maxH);
}
int main(){run("source_null_reduced",original::fixedcpu_geodesic,original::fixedcpu_rk5);run("canonical_schwarzschild",canonical::fixedcpu_geodesic,canonical::fixedcpu_rk5);}
