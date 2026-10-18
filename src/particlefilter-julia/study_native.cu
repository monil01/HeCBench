
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <chrono>
#include <unistd.h>
#define BLOCK_SIZE 256
#define PI 3.1415926535897932f
#define A 1103515245
#define C 12345
#define M INT_MAX
#define SCALE_FACTOR 300.f
#include "../particlefilter-cuda/kernel_likelihood.h"
#include "../particlefilter-cuda/kernel_sum.h"
#include "native_corrected/kernel_normalize_weights.h"
#include "../particlefilter-cuda/kernel_find_index.h"
#include "study_cases.h"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
const int G=16;const float GUARD=12345.5f;
template<class T>std::vector<T>readv(const char*n){std::ifstream f(std::string("/home/imo/HeCBench_julia/src/particlefilter-julia/")+n,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("fixture open");size_t z=f.tellg();f.seekg(0);std::vector<T>v(z/sizeof(T));if(!f.read((char*)v.data(),z))throw std::runtime_error("short fixture");return v;}
template<class T>T*upload(std::vector<T>&h){T*p;CK(cudaMalloc(&p,h.size()*sizeof(T)));CK(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
struct State{int n,nb;std::vector<std::vector<float>>h;std::vector<float*>d;std::vector<int>hind,hseed,hobj;int*ind,*seed,*obj;std::vector<unsigned char>hi;unsigned char*im;};
void frame(State&s,int k){int n=s.n,b=s.nb;kernel_likelihood<<<b,256>>>(s.d[0]+G,s.d[1]+G,s.d[6]+G,s.d[7]+G,s.ind+G,s.obj+G,s.d[2]+G,s.im+G,s.d[3]+G,s.seed+G,s.d[8]+G,n,69,64,4,k,16384);kernel_sum<<<1,1>>>(s.d[8]+G,n);kernel_scale_weights<<<b,256>>>(s.d[3]+G,s.d[8]+G,n);kernel_normalize_weights<<<1,256>>>(s.d[3]+G,s.d[8]+G,s.d[4]+G,s.d[5]+G,s.seed+G,n);kernel_find_index<<<b,256>>>(s.d[0]+G,s.d[1]+G,s.d[4]+G,s.d[5]+G,s.d[6]+G,s.d[7]+G,n);CK(cudaDeviceSynchronize());}
void reset(State&s){for(int j=0;j<9;j++)CK(cudaMemcpy(s.d[j],s.h[j].data(),s.h[j].size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(s.ind,s.hind.data(),s.hind.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(s.seed,s.hseed.data(),s.hseed.size()*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());}
int main(){auto video=readv<unsigned char>("video_u8.bin");auto offsets=readv<int>("offsets_i32.bin");auto gold=readv<double>("oracle_f64.bin");std::vector<State>states;std::vector<double>prefix;
for(auto c:cases){State s;s.n=c.n;s.nb=(c.n+255)/256;for(int j=0;j<9;j++){int len=j==8?s.nb:s.n;std::vector<float>h(len+2*G,GUARD);if(j==6||j==7)for(int i=0;i<s.n;i++)h[i+G]=j==6?32.125f+(i%7)/16.f:31.375f-(i%5)/32.f;if(j==3)for(int i=0;i<s.n;i++)h[i+G]=1.f/s.n;s.h.push_back(h);s.d.push_back(upload(h));}s.hind.assign(69*s.n+2*G,1234567);s.hseed.assign(s.n+2*G,1234567);for(int i=0;i<s.n;i++)s.hseed[i+G]=i+1;s.hobj.assign(offsets.size()+2*G,1234567);std::copy(offsets.begin(),offsets.end(),s.hobj.begin()+G);s.hi.assign(video.size()+2*G,251);std::copy(video.begin(),video.end(),s.hi.begin()+G);s.ind=upload(s.hind);s.seed=upload(s.hseed);s.obj=upload(s.hobj);s.im=upload(s.hi);for(int j=0;j<9;j++){std::vector<float>v(s.h[j].size());CK(cudaMemcpy(v.data(),s.d[j],v.size()*4,cudaMemcpyDeviceToHost));prefix.insert(prefix.end(),v.begin(),v.end());}for(auto p:std::vector<std::pair<int*,std::vector<int>*>>{{s.ind,&s.hind},{s.seed,&s.hseed},{s.obj,&s.hobj}}){std::vector<int>v(p.second->size());CK(cudaMemcpy(v.data(),p.first,v.size()*4,cudaMemcpyDeviceToHost));prefix.insert(prefix.end(),v.begin(),v.end());}std::vector<unsigned char>vi(s.hi.size());CK(cudaMemcpy(vi.data(),s.im,vi.size(),cudaMemcpyDeviceToHost));prefix.insert(prefix.end(),vi.begin(),vi.end());states.push_back(s);}
for(auto&s:states){for(int k=1;k<4;k++)frame(s,k);reset(s);}
for(int trial=1;trial<=5;trial++){std::vector<double>out,times;bool ok=true;double maxerr=0;int checked=0;for(int c=0;c<3;c++){auto&s=states[c];reset(s);for(int k=1;k<4;k++){auto a=std::chrono::steady_clock::now();frame(s,k);auto z=std::chrono::steady_clock::now();times.push_back(std::chrono::duration<double>(z-a).count());int off=cases[c].frames[k-1].off;for(int j=0;j<9;j++){std::vector<float>v(s.h[j].size());CK(cudaMemcpy(v.data(),s.d[j],v.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<(int)v.size();i++){if(i<G||i>=(int)v.size()-G)ok&=v[i]==GUARD;else{double ref=gold[off++],e=fabs(v[i]-ref);ok&=std::isfinite(v[i])&&e<=1e-4+3e-5*fabs(ref);maxerr=fmax(maxerr,e);checked++;}}out.insert(out.end(),v.begin(),v.end());}for(auto p:std::vector<std::pair<int*,std::vector<int>*>>{{s.ind,&s.hind},{s.seed,&s.hseed}}){std::vector<int>v(p.second->size());CK(cudaMemcpy(v.data(),p.first,v.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<(int)v.size();i++)if(i<G||i>=(int)v.size()-G)ok&=v[i]==1234567;else{ok&=v[i]==gold[off++];checked++;}out.insert(out.end(),v.begin(),v.end());}std::vector<int>ob(s.hobj.size());std::vector<unsigned char>vi(s.hi.size());CK(cudaMemcpy(ob.data(),s.obj,ob.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(vi.data(),s.im,vi.size(),cudaMemcpyDeviceToHost));ok&=ob==s.hobj&&vi==s.hi;out.insert(out.end(),ob.begin(),ob.end());out.insert(out.end(),vi.begin(),vi.end());}}
std::string path="/tmp/particlefilter_"+std::to_string(getpid())+"_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)prefix.data(),prefix.size()*8);f.write((char*)out.data(),out.size()*8);f.close();double sum=0;for(auto x:times)sum+=x;printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"timing_region\":\"sum_nine_frame_five_kernel_pipelines_and_device_completion\",\"max_abs_error\":%.12g,\"max_rel_error\":null,\"exact_integer_and_input_guard_checks\":true,\"checked_values\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"}}\n",trial,ok?"true":"false",sum,maxerr,checked,path.c_str(),prefix.size()+out.size(),prefix.size()+out.size());}}
