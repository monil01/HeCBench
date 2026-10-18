#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "../reaction-cuda/kernels.cu"
#ifndef FIXTURE
#error FIXTURE required
#endif
#ifndef STUDYDIR
#error STUDYDIR required
#endif
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
std::vector<double>load(const std::string&name){std::ifstream f(name,std::ios::binary|std::ios::ate);if(!f)exit(3);auto n=f.tellg();if(n<0||n%8!=0)exit(3);f.seekg(0);std::vector<double>a((size_t)n/8);f.read((char*)a.data(),n);if(!f)exit(3);return a;}
struct Buffer{float*base,*p;std::vector<float>initial;Buffer(std::vector<float>h):initial(h){CK(cudaMalloc(&base,(h.size()+16)*4));p=base+8;reset();}void reset(){std::vector<float>h(initial.size()+16,-12345.f);std::copy(initial.begin(),initial.end(),h.begin()+8);CK(cudaMemcpy(base,h.data(),h.size()*4,cudaMemcpyHostToDevice));}std::vector<float>get(){std::vector<float>a(initial.size());CK(cudaMemcpy(a.data(),p,a.size()*4,cudaMemcpyDeviceToHost));return a;}bool guards(std::vector<double>&out){float a[8],b[8];CK(cudaMemcpy(a,base,32,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b,p+initial.size(),32,cudaMemcpyDeviceToHost));bool ok=true;for(float x:a){ok&=x==-12345.f;out.push_back(x);}for(float x:b){ok&=x==-12345.f;out.push_back(x);}return ok;}};
struct Case{unsigned M,N,steps,ci;float ca,cb,dt,c1,c2;std::vector<Buffer*>d;std::vector<double>gold;Case(unsigned m,unsigned s,unsigned c,std::vector<float>par,std::vector<float>a,std::vector<float>b):M(m),N(m*m*m),steps(s),ci(c),ca(par[0]),cb(par[1]),dt(par[2]),c1(par[3]),c2(par[4]){d.push_back(new Buffer(a));d.push_back(new Buffer(b));for(int j=2;j<9;++j)d.push_back(new Buffer(std::vector<float>(N,0)));gold=load(std::string(STUDYDIR)+"/oracle_case"+std::to_string(ci)+".bin");if(gold.size()!=9*N)exit(3);}};
struct Outcome{bool ok;std::vector<double>art;double time,ma,mda,mdb;};
Outcome execute(Case&c){for(auto a:c.d)a->reset();CK(cudaDeviceSynchronize());auto ina=c.d[0]->get(),inb=c.d[1]->get();Outcome o;o.ok=ina==c.d[0]->initial&&inb==c.d[1]->initial;o.ma=0;
 auto a=c.d[0]->p,b=c.d[1]->p,dx=c.d[2]->p,dy=c.d[3]->p,dz=c.d[4]->p,ra=c.d[5]->p,rb=c.d[6]->p,da=c.d[7]->p,db=c.d[8]->p;dim3 grid(c.M/2,c.M),tx(c.M,2),ty(2,c.M);unsigned sh=2*c.M*4;
 auto start=std::chrono::steady_clock::now();
 for(unsigned step=0;step<c.steps;++step){
  derivative_x2_zeroflux<<<grid,tx,sh>>>(a,dx,c.M,c.M);derivative_y2_zeroflux<<<grid,ty,sh>>>(a,dy,c.M,c.M,2);derivative_z2_zeroflux<<<grid,ty,sh>>>(a,dz,c.M,c.M,c.M,2);construct_laplacian<<<c.N/c.M,c.M>>>(da,dx,dy,dz,c.N,c.ca);
  derivative_x2_zeroflux<<<grid,tx,sh>>>(b,dx,c.M,c.M);derivative_y2_zeroflux<<<grid,ty,sh>>>(b,dy,c.M,c.M,2);derivative_z2_zeroflux<<<grid,ty,sh>>>(b,dz,c.M,c.M,c.M,2);construct_laplacian<<<c.N/c.M,c.M>>>(db,dx,dy,dz,c.N,c.cb);reaction_gray_scott<<<c.N/c.M,c.M>>>(a,b,ra,rb,c.N,c.c1,c.c2);update<<<c.N/c.M,c.M>>>(a,b,da,db,ra,rb,c.N,c.dt);
 }
 CK(cudaDeviceSynchronize());o.time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();size_t gi=0;std::vector<std::vector<float>>fields;
 for(auto buf:c.d){auto h=buf->get();fields.push_back(h);for(float v:h){double ref=c.gold[gi++],err=fabs((double)v-ref);o.ma=std::max(o.ma,err);o.ok&=std::isfinite(v)&&err<=1e-6+5e-6*fabs(ref);if(c.ci==3)o.ok&=(double)v==ref;o.art.push_back(v);}}
 o.ok&=gi==c.gold.size();for(unsigned i=0;i<c.N;++i)o.ok&=fields[0][i]>=0&&fields[1][i]>=0;double sda=0,sdb=0;for(unsigned i=0;i<c.N;++i){sda+=fields[7][i];sdb+=fields[8][i];}o.mda=sda/c.N;o.mdb=sdb/c.N;o.ok&=fabs(o.mda)<=1e-6&&fabs(o.mdb)<=1e-6;
 if(c.ci==4){double before=0,after=0;for(unsigned i=0;i<c.N;++i){before+=ina[i];after+=fields[0][i];o.ok&=fields[1][i]==0;}o.ok&=fabs((after-before)/c.N)<=1e-6;}
 for(float v:ina)o.art.push_back(v);for(float v:inb)o.art.push_back(v);for(auto buf:c.d)o.ok&=buf->guards(o.art);return o;
}
int main(){auto pre=load(FIXTURE);if(pre.size()!=90141||pre[0]!=4)return 3;size_t pos=1;std::vector<Case*>cs;for(unsigned ci=1;ci<=4;++ci){unsigned M=pre[pos++],steps=pre[pos++],N=M*M*M;std::vector<float>par(5),a(N),b(N);for(float&v:par)v=pre[pos++];for(float&v:a)v=pre[pos++];for(float&v:b)v=pre[pos++];cs.push_back(new Case(M,steps,ci,par,a,b));}if(pos!=pre.size())return 3;for(auto c:cs)execute(*c);
 for(int trial=1;trial<=5;++trial){auto art=pre;std::vector<double>times,mas,means;double total=0,ma=0;bool ok=true;for(auto c:cs){auto o=execute(*c);ok&=o.ok;art.insert(art.end(),o.art.begin(),o.art.end());times.push_back(o.time);total+=o.time;mas.push_back(o.ma);ma=std::max(ma,o.ma);means.push_back(o.mda);means.push_back(o.mdb);}std::string f="output_trial"+std::to_string(trial)+".bin";std::ofstream out(f,std::ios::binary);out.write((char*)art.data(),art.size()*8);out.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum4_complete_zeroflux_GrayScott_timestep_batches_130_launches\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"path_times_seconds\":[",trial,total,ok?"true":"false",art.size(),ma,f.c_str(),art.size());for(size_t i=0;i<times.size();++i)printf("%s%.17g",i?",":"",times[i]);printf("],\"path_labels\":[\"nonuniform16cubed5steps\",\"central32cubed4steps\",\"equilibrium16cubed3steps\",\"diffusion16cubed1step\"],\"case_max_abs_errors\":[");for(size_t i=0;i<mas.size();++i)printf("%s%.17g",i?",":"",mas[i]);printf("],\"mean_diffusion_A_B_eachcase\":[");for(size_t i=0;i<means.size();++i)printf("%s%.17g",i?",":"",means[i]);printf("]}\n");
 }
 return 0;
}
