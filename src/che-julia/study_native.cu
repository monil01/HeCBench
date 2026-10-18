#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <string>
#include <algorithm>
#define DATAXSIZE 16
#define DATAYSIZE 8
#define DATAZSIZE 8
namespace A{
#include "native_kernels.inc"
}
#undef DATAXSIZE
#undef DATAYSIZE
#undef DATAZSIZE
void stepA(std::vector<double*>&a){using namespace A;using P=double(*)[8][16];dim3 block(16,4,4),grid((16+15)/16,(8+3)/4,(8+3)/4);chemicalPotential<<<grid,block>>>((P)a[0],(P)a[2],1.,1.,1.,.5,-2./9.,-2./9.,2./9.);localFreeEnergyFunctional<<<grid,block>>>((P)a[0],(P)a[3],1.,1.,1.,.5,-2./9.,-2./9.,2./9.);cahnHilliard<<<grid,block>>>((P)a[1],(P)a[0],(P)a[2],1.,.01,1.,1.,1.);Swap<<<grid,block>>>((P)a[1],(P)a[0]);}
#define DATAXSIZE 19
#define DATAYSIZE 9
#define DATAZSIZE 7
namespace B{
#include "native_kernels.inc"
}
#undef DATAXSIZE
#undef DATAYSIZE
#undef DATAZSIZE
void stepB(std::vector<double*>&a){using namespace B;using P=double(*)[9][19];dim3 block(16,4,4),grid((19+15)/16,(9+3)/4,(7+3)/4);chemicalPotential<<<grid,block>>>((P)a[0],(P)a[2],1.,1.,1.,.5,-2./9.,-2./9.,2./9.);localFreeEnergyFunctional<<<grid,block>>>((P)a[0],(P)a[3],1.,1.,1.,.5,-2./9.,-2./9.,2./9.);cahnHilliard<<<grid,block>>>((P)a[1],(P)a[0],(P)a[2],1.,.01,1.,1.,1.);Swap<<<grid,block>>>((P)a[1],(P)a[0]);}
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
std::vector<double>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);size_t n=f.tellg();f.seekg(0);std::vector<double>v(n/8);f.read((char*)v.data(),n);return v;}
struct C{int nx,ny,nz,steps,mode;};C cs[]={{16,8,8,1,0},{19,9,7,3,0},{16,8,8,3,1},{16,8,8,3,2}};struct S{std::vector<double*>raw,a;};
int main(){auto fixture=load(CHE_FIXTURE),oracle=load(CHE_ORACLE);std::vector<S>s(4);for(int c=0;c<4;c++)for(int k=0;k<4;k++){double*p;int v=cs[c].nx*cs[c].ny*cs[c].nz;ck(cudaMalloc(&p,(v+2)*8));s[c].raw.push_back(p);s[c].a.push_back(p+1);}
auto run=[&](int c){int v=cs[c].nx*cs[c].ny*cs[c].nz;size_t off=0;for(int j=0;j<c;j++)off+=cs[j].nx*cs[j].ny*cs[j].nz;std::vector<double>h(v+2,-8192.25);for(int k=0;k<4;k++){std::fill(h.begin()+1,h.end()-1,-8192.25);if(k==0)std::copy(fixture.begin()+off,fixture.begin()+off+v,h.begin()+1);ck(cudaMemcpy(s[c].raw[k],h.data(),(v+2)*8,cudaMemcpyHostToDevice));}ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int st=0;st<cs[c].steps;st++){if(cs[c].nx==16)stepA(s[c].a);else stepB(s[c].a);}ck(cudaGetLastError());ck(cudaDeviceSynchronize());return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};for(int c=0;c<4;c++)run(c);for(int trial=1;trial<=5;trial++){auto out=fixture;size_t oi=0,fi=0;double elapsed=0,err=0,maxmass=0;bool ok=true;for(int c=0;c<4;c++){elapsed+=run(c);int v=cs[c].nx*cs[c].ny*cs[c].nz;std::vector<double>guards;double mass=0,initialmass=0;for(int i=0;i<v;i++)initialmass+=fixture[fi+i];fi+=v;for(int k=0;k<4;k++){std::vector<double>h(v+2);ck(cudaMemcpy(h.data(),s[c].raw[k],(v+2)*8,cudaMemcpyDeviceToHost));ok&=h.front()==-8192.25&&h.back()==-8192.25;guards.push_back(h.front());guards.push_back(h.back());for(int i=0;i<v;i++){double e=std::abs(h[i+1]-oracle[oi]);err=std::max(err,e);ok&=std::isfinite(h[i+1])&&e<=1e-11+1e-11*std::abs(oracle[oi]);oi++;if(k==0)mass+=h[i+1];}out.insert(out.end(),h.begin()+1,h.end()-1);}maxmass=std::max(maxmass,std::abs(mass-initialmass));ok&=std::abs(mass-initialmass)<=1e-10;out.insert(out.end(),guards.begin(),guards.end());}std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();printf("STUDY_MASS_DRIFT %.17g\n",maxmass);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"timing_region\":\"sum4_preallocated_periodic_fourlaunch_step_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,elapsed,ok?"true":"false",out.size(),err,path.c_str(),out.size());if(!ok)return 3;}for(auto&t:s)for(auto*q:t.raw)ck(cudaFree(q));}
