
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "../sampling-cuda/kernels.cu"
#include "study_cases.h"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
const int G=16;const float GUARD=12345.5f;const int IGUARD=1234567;
template<class T>std::vector<T>readv(const char*n){std::ifstream f(std::string("/home/imo/HeCBench_julia/src/sampling-julia/")+n,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("fixtureopen");size_t sz=f.tellg();if(sz%sizeof(T))throw std::runtime_error("fixturedtype");f.seekg(0);std::vector<T>v(sz/sizeof(T));if(!f.read((char*)v.data(),sz))throw std::runtime_error("shortfixture");return v;}
template<class T>T*upload(std::vector<T>&h){T*p;CK(cudaMalloc(&p,h.size()*sizeof(T)));CK(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T>std::vector<T>get(T*p,int n){std::vector<T>v(n);CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
struct State{Case c;int xn,dn;std::vector<float>hx,hb,ho,hd;std::vector<int>hn;float*x,*b,*o,*data;int*ns;};
void reset(State&s){CK(cudaMemcpy(s.x,s.hx.data(),s.hx.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(s.data,s.hd.data(),s.hd.size()*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());}
double run(State&s){double ns=0;auto c=s.c;kernel_dataset<float,int>(s.x+G,c.exact+c.sampled,c.cols,s.b+G,c.bg,s.data+G,s.o+G,s.ns+G,c.sampled,c.kmax,c.seed,ns);CK(cudaGetLastError());return ns*1e-9;}
int main(){auto masks=readv<unsigned char>("oracle_masks_u8.bin");auto features=readv<float>("features_f32.bin");std::vector<State>states;std::vector<float>prefix;
for(auto c:cases){State s;s.c=c;s.xn=(c.exact+c.sampled)*c.cols;s.dn=s.xn*c.bg;s.hx.assign(s.xn+32,GUARD);std::fill(s.hx.begin()+G,s.hx.end()-G,0.f);for(int i=0;i<c.exact;i++){s.hx[i*c.cols+i+G]=1;s.hx[i*c.cols+i+1+G]=1;}s.hb.assign(c.bg*c.cols+32,GUARD);std::copy(features.begin()+c.fo,features.begin()+c.fo+c.bg*c.cols,s.hb.begin()+G);s.ho.assign(c.cols+32,GUARD);std::copy(features.begin()+c.fo+c.bg*c.cols,features.begin()+c.fo+(c.bg+1)*c.cols,s.ho.begin()+G);s.hn.assign(c.sampled/2+32,IGUARD);for(int i=0;i<c.sampled/2;i++)s.hn[i+G]=c.kmax-i%2;s.hd.assign(s.dn+32,GUARD);s.x=upload(s.hx);s.b=upload(s.hb);s.o=upload(s.ho);s.ns=upload(s.hn);s.data=upload(s.hd);for(auto p:std::vector<std::pair<float*,int>>{{s.x,(int)s.hx.size()},{s.b,(int)s.hb.size()},{s.o,(int)s.ho.size()}}){auto v=get(p.first,p.second);prefix.insert(prefix.end(),v.begin(),v.end());}auto vn=get(s.ns,(int)s.hn.size());prefix.insert(prefix.end(),vn.begin(),vn.end());auto lo=get(s.data,G),hi=get(s.data+G+s.dn,G);prefix.insert(prefix.end(),lo.begin(),lo.end());prefix.insert(prefix.end(),hi.begin(),hi.end());for(int j=0;j<4;j++)prefix.push_back((c.seed>>(16*j))&65535);run(s);reset(s);states.push_back(std::move(s));}
for(int trial=1;trial<=5;trial++){std::vector<float>art=prefix;art.reserve(96148239);bool ok=true;double seconds=0;for(auto&s:states){auto c=s.c;reset(s);seconds+=run(s);auto x=get(s.x,(int)s.hx.size()),data=get(s.data,(int)s.hd.size()),b=get(s.b,(int)s.hb.size()),o=get(s.o,(int)s.ho.size());auto ns=get(s.ns,(int)s.hn.size());ok&=b==s.hb&&o==s.ho&&ns==s.hn;for(int i=0;i<s.xn+32;i++)ok&=x[i]==(i>=G&&i<G+s.xn?float(masks[c.mo+i-G]):GUARD);for(int i=0;i<s.dn+32;i++){if(i<G||i>=s.dn+G)ok&=data[i]==GUARD;else{int index=i-G,col=index%c.cols,row=index/(c.bg*c.cols),bg=(index/c.cols)%c.bg;float expected=masks[c.mo+row*c.cols+col]?s.ho[col+G]:s.hb[bg*c.cols+col+G];ok&=data[i]==expected;}}for(auto*v:{&x,&data,&b,&o})art.insert(art.end(),v->begin(),v->end());art.insert(art.end(),ns.begin(),ns.end());}
std::string path="sampling_trial"+std::to_string(trial)+".bin";std::ofstream(path,std::ios::binary).write((char*)art.data(),art.size()*4);printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"timing_region\":\"sum_seven_source_exact_and_concurrent_sample_scatter_pipelines_plus_completion\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"}}\n",trial,ok?"true":"false",seconds,path.c_str(),art.size(),art.size());}}
