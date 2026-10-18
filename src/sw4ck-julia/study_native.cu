#include "../sw4ck-cuda/curvilinear4sg.cu"
#include "study_cpu_native.h"
#undef ni
#undef nij
#undef nijk
#undef base
#undef base3
#undef base4
#undef ifirst0
#undef jfirst0
#undef mu
#undef la
#undef jac
#undef u
#undef lu
#undef met
#undef strx
#undef stry
#undef acof
#undef bope
#undef ghcof
#undef acof_no_gp
#undef ghcof_no_gp
#undef i6
#undef tf
#undef c1
#undef c2
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <tuple>
#include <cmath>
struct Case{int ni,nj,nk,mode,n;std::vector<double>u,mu,la,met,jac,init,sx,sy,cof,ref;double *ud,*mud,*lad,*metd,*jd,*out,*xd,*yd,*cd;};
void cpu_pipeline(Case&c){int fi=-2,li=c.ni-3,fj=-2,lj=c.nj-3,fk=-2,lk=c.nk+3;auto ac=c.cof.data()+468,bo=c.cof.data()+414,gh=c.cof.data()+852;for(int stage=1;stage<=5;stage++){int ks=stage==1?1:stage==5?c.nk-5:7,ke=stage==1?7:stage==5?c.nk+1:c.nk-5;for(int k=ks;k<ke;k++)for(int j=0;j<lj-1;j++)for(int i=0;i<li-1;i++){
#define CPUARGS i,li-1,j,lj-1,k,ke,fi,li,fj,lj,fk,lk
#define ARRAYARGS c.u.data(),c.mu.data(),c.la.data(),c.met.data(),c.jac.data(),c.ref.data(),ac,bo,gh,ac,gh,c.sx.data(),c.sy.data()
if(stage==1)cpu_kernel1(CPUARGS,1.,-1.,ARRAYARGS);if(stage==2)cpu_kernel2(CPUARGS,1.,-1.,ARRAYARGS);if(stage==3)cpu_kernel3(CPUARGS,1.,-1.,ARRAYARGS);if(stage==4)cpu_kernel4(CPUARGS,1.,-1.,ARRAYARGS);if(stage==5)cpu_kernel5(CPUARGS,c.nk,1.,-1.,ARRAYARGS);
#undef CPUARGS
#undef ARRAYARGS
}}}
bool physical(const Case&c,const std::vector<double>&out){if(c.mode==2)return true;double vals[]={64,73,82};for(int z=7;z<=c.nk-6;z++)for(int y=0;y<=c.nj-5;y++)for(int x=0;x<=c.ni-5;x++)for(int a=0;a<3;a++){int q=x+2+c.ni*(y+2+c.nj*(z+2))+a*c.n;if(fabs(out[q]-(c.init[q]-(c.mode==0?vals[a]:0)))>1e-8)return false;}return true;}
int main(){std::vector<Case>cases;std::vector<double>prefix;for(int shape=0;shape<2;shape++)for(int mode=0;mode<3;mode++){Case c;c.ni=shape?19:13;c.nj=shape?13:11;c.nk=shape?20:18;c.mode=mode;c.n=c.ni*c.nj*(c.nk+6);c.u.assign(3*c.n,0);c.mu.assign(c.n,2);c.la.assign(c.n,3);c.met.assign(4*c.n,0);c.jac.assign(c.n,1);c.init.resize(3*c.n);c.sx.assign(c.ni,1);c.sy.assign(c.nj,1);c.cof.resize(864);for(int q=0;q<c.n;q++){c.met[q]=c.met[3*c.n+q]=1;}for(int q=0;q<3*c.n;q++)c.init[q]=double((q*7)%31-15)/32;for(int q=0;q<864;q++)c.cof[q]=double(q)/1000;for(int iz=0;iz<c.nk+6;iz++)for(int iy=0;iy<c.nj;iy++)for(int ix=0;ix<c.ni;ix++){int q=ix+c.ni*(iy+c.nj*iz),x=ix-2,y=iy-2,z=iz-2;if(mode==0){c.u[q]=x*x+2*x*y+3*x*z;c.u[c.n+q]=2*y*y+4*x*y+5*y*z;c.u[2*c.n+q]=3*z*z+6*x*z+7*y*z;}else if(mode==1){c.u[q]=1;c.u[c.n+q]=2;c.u[2*c.n+q]=3;}else{for(int a=0;a<3;a++)c.u[a*c.n+q]=double((q*17+a*13)%97-48)/16;c.mu[q]+=double(q%7)/32;c.la[q]+=double(q%5)/32;c.jac[q]+=double(q%13)/64;c.met[q]+=double(q%5)/64;c.met[c.n+q]=.125+double(q%3)/128;c.met[2*c.n+q]=.0625+double(q%7)/128;c.met[3*c.n+q]=.75+double(q%11)/128;}}if(mode==2){for(int q=0;q<c.ni;q++)c.sx[q]=1+double(q)/64;for(int q=0;q<c.nj;q++)c.sy[q]=1+double(q)/64;}c.ref=c.init;cpu_pipeline(c);
#define ALLOC(ptr,vec) cudaMalloc(&c.ptr,c.vec.size()*8);cudaMemcpy(c.ptr,c.vec.data(),c.vec.size()*8,cudaMemcpyHostToDevice)
ALLOC(ud,u);ALLOC(mud,mu);ALLOC(lad,la);ALLOC(metd,met);ALLOC(jd,jac);ALLOC(out,init);ALLOC(xd,sx);ALLOC(yd,sy);ALLOC(cd,cof);
#undef ALLOC
for(int q:{c.ni,c.nj,c.nk,mode,3,-2,c.ni-3,-2,c.nj-3,-2,c.nk+3})prefix.push_back(q);for(auto*v:{&c.u,&c.mu,&c.la,&c.met,&c.jac,&c.init,&c.sx,&c.sy,&c.cof})prefix.insert(prefix.end(),v->begin(),v->end());cases.push_back(c);}
auto run=[&](Case&c,std::vector<double>&all){int side[]={0,0,0,0,1,1};double time=0;for(int rep=0;rep<3;rep++){cudaMemcpy(c.out,c.init.data(),c.init.size()*8,cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();curvilinear4sg_ci(-2,c.ni-3,-2,c.nj-3,-2,c.nk+3,c.ud,c.mud,c.lad,c.metd,c.jd,c.out,side,c.cd+468,c.cd+414,c.cd+852,c.cd+468,c.cd+852,c.xd,c.yd,c.nk,'-');cudaDeviceSynchronize();time+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/3;}std::vector<double>out(c.init.size());cudaMemcpy(out.data(),c.out,out.size()*8,cudaMemcpyDeviceToHost);bool ok=cudaGetLastError()==cudaSuccess;double err=0;for(size_t i=0;i<out.size();i++){double e=fabs(out[i]-c.ref[i]);err=std::max(err,e);ok&=std::isfinite(out[i])&&e<=1e-8+1e-10*fabs(c.ref[i]);}ok&=physical(c,out);all.insert(all.end(),out.begin(),out.end());return std::make_tuple(time,ok,err);};std::vector<double>tmp;for(auto&c:cases)run(c,tmp);for(int trial=1;trial<=5;trial++){auto all=prefix;double time=0,err=0;bool ok=true;std::cout<<"STUDY_COMPONENT_SECONDS ";for(auto&c:cases){auto[t,pass,e]=run(c,all);time+=t;ok&=pass;err=std::max(err,e);std::cout<<std::setprecision(17)<<t<<",";}std::cout<<std::endl;std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)all.data(),all.size()*8);f.close();std::cout<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<time<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<all.size()<<",\"max_abs_error\":"<<err<<",\"timing_region\":\"sum6_three_original_fivekernel_pipeline_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"float64\",\"count\":"<<all.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}}
