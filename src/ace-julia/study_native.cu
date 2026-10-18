#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <string>
#define DATAXSIZE 16
#define DATAYSIZE 16
#define DATAZSIZE 16
namespace Ace16 {
#include "native_kernels.inc"
}
#undef DATAXSIZE
#undef DATAYSIZE
#undef DATAZSIZE
#define DATAXSIZE 24
#define DATAYSIZE 24
#define DATAZSIZE 24
namespace Ace24 {
#include "native_kernels.inc"
}
#undef DATAXSIZE
#undef DATAYSIZE
#undef DATAZSIZE
struct Case{int n,steps,mode;}; const Case cases[]={{16,1,0},{16,3,0},{24,3,1},{16,3,2}};
std::vector<double> load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);std::vector<double> v(n/8);f.read((char*)v.data(),n);return v;}
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(2);}}
void step16(std::vector<double*>&a){using namespace Ace16;using A=double(*)[16][16];dim3 b(8,8,4),g(16/8,16/8,16/4);double h=.4,ep=.07,w=1.,t=1.25/std::sqrt(2.)*.64,la=1.25/std::sqrt(2.)/.5;
calculateForce<<<g,b>>>((A)a[0],(A)a[4],(A)a[5],(A)a[6],h,h,h,ep,w,t);
allenCahn<<<g,b>>>((A)a[1],(A)a[0],(A)a[2],(A)a[4],(A)a[5],(A)a[6],ep,w,t,la,.01,h,h,h);
boundaryConditionsPhi<<<g,b>>>((A)a[1]);
thermalEquation<<<g,b>>>((A)a[3],(A)a[2],(A)a[1],(A)a[0],2.,.01,h,h,h);
boundaryConditionsU<<<g,b>>>((A)a[3],.8);
swapGrid<<<g,b>>>((A)a[1],(A)a[0]);
swapGrid<<<g,b>>>((A)a[3],(A)a[2]);}
void step24(std::vector<double*>&a){using namespace Ace24;using A=double(*)[24][24];dim3 b(8,8,4),g(24/8,24/8,24/4);double h=.4,ep=.07,w=1.,t=1.25/std::sqrt(2.)*.64,la=1.25/std::sqrt(2.)/.5;
calculateForce<<<g,b>>>((A)a[0],(A)a[4],(A)a[5],(A)a[6],h,h,h,ep,w,t);
allenCahn<<<g,b>>>((A)a[1],(A)a[0],(A)a[2],(A)a[4],(A)a[5],(A)a[6],ep,w,t,la,.01,h,h,h);
boundaryConditionsPhi<<<g,b>>>((A)a[1]);
thermalEquation<<<g,b>>>((A)a[3],(A)a[2],(A)a[1],(A)a[0],2.,.01,h,h,h);
boundaryConditionsU<<<g,b>>>((A)a[3],.8);
swapGrid<<<g,b>>>((A)a[1],(A)a[0]);
swapGrid<<<g,b>>>((A)a[3],(A)a[2]);}
int main(){auto fixture=load(ACE_FIXTURE),oracle=load(ACE_ORACLE);struct State{std::vector<double*>raw,a;};std::vector<State>s(4);size_t fi=0;
for(int c=0;c<4;c++){int v=cases[c].n*cases[c].n*cases[c].n;for(int k=0;k<7;k++){double*p;ck(cudaMalloc(&p,(v+2)*8));s[c].raw.push_back(p);s[c].a.push_back(p+1);}}
auto run=[&](int c){int v=cases[c].n*cases[c].n*cases[c].n;size_t off=0;for(int j=0;j<c;j++)off+=2*cases[j].n*cases[j].n*cases[j].n;std::vector<double>reset(v+2,-8192.25);for(int k=0;k<7;k++){reset[0]=reset[v+1]=-1048576.5;std::fill(reset.begin()+1,reset.end()-1,-8192.25);if(k==0||k==2)std::copy(fixture.begin()+off+(k==2?v:0),fixture.begin()+off+(k==2?2*v:v),reset.begin()+1);ck(cudaMemcpy(s[c].raw[k],reset.data(),(v+2)*8,cudaMemcpyHostToDevice));}ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int st=0;st<cases[c].steps;st++){if(cases[c].n==16)step16(s[c].a);else step24(s[c].a);}ck(cudaGetLastError());ck(cudaDeviceSynchronize());return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
for(int c=0;c<4;c++)run(c);for(int trial=1;trial<=5;trial++){auto out=fixture;size_t oi=0;double time=0,err=0;bool ok=true;for(int c=0;c<4;c++){time+=run(c);int v=cases[c].n*cases[c].n*cases[c].n;for(int k=0;k<7;k++){std::vector<double>h(v+2);ck(cudaMemcpy(h.data(),s[c].raw[k],(v+2)*8,cudaMemcpyDeviceToHost));ok&=h.front()==-1048576.5&&h.back()==-1048576.5;for(int i=0;i<v;i++){double e=std::abs(h[i+1]-oracle[oi++]);err=std::max(err,e);ok&=std::isfinite(h[i+1])&&e<=1e-10+1e-10*std::abs(oracle[oi-1]);}out.insert(out.end(),h.begin()+1,h.end()-1);}}std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"timing_region\":\"sum4_preallocated_source_sevenlaunch_steps_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,time,ok?"true":"false",out.size(),err,path.c_str(),out.size());if(!ok)return 3;}for(auto&t:s)for(auto*p:t.raw)ck(cudaFree(p));}
