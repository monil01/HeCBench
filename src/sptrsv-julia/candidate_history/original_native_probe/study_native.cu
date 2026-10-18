// Original custom solver body in study_original_kernel.cuh; MIT LICENSE.source.
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <cstdio>
#include "study_original_kernel.cuh"
#include "study_cases.h"
#define CK(x) do{if((x)!=cudaSuccess)throw std::runtime_error(#x);}while(0)
const int G=8,R=3;const double GUARD=123456789.;
std::vector<double>readv(const char*n){std::ifstream f(n,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("openfixture");size_t z=f.tellg();if(z%8)throw std::runtime_error("dtype");f.seekg(0);std::vector<double>a(z/8);if(!f.read((char*)a.data(),z))throw std::runtime_error("shortfixture");return a;}
template<class T>struct Buf{std::vector<T>h;T*d;Buf(std::vector<double>&f,int off,int n,bool zero=false){h.assign(n+2*G,T(GUARD));for(int i=0;i<n;i++)h[G+i]=zero?0:T(f.at(off+i));CK(cudaMalloc(&d,h.size()*sizeof(T)));reset();}void reset(){CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){auto q=h;CK(cudaMemcpy(q.data(),d,q.size()*sizeof(T),cudaMemcpyDeviceToHost));return q;}T*p(){return d+G;}};
struct State{Case c;Buf<int>rp,ci,map,flags;Buf<double>v,b,x;State(Case c,std::vector<double>&f):c(c),rp(f,c.io,c.m+1),ci(f,c.io+c.m+1,c.nnz),map(f,c.io+c.m+1+2*c.nnz+c.m,c.length),flags(f,0,c.m,true),v(f,c.io+c.m+1+c.nnz,c.nnz),b(f,c.io+c.m+1+2*c.nnz,c.m),x(f,0,c.m,true){}};
template<class T>void append(std::vector<double>&a,std::vector<T>q){for(auto x:q)a.push_back(double(x));}
void capture(State&s,std::vector<double>&a){append(a,s.rp.get());append(a,s.ci.get());append(a,s.v.get());append(a,s.b.get());append(a,s.map.get());append(a,s.x.get());append(a,s.flags.get());}
void launch(State&s){sptrsv_mix<<<(s.c.length-2+8)/8,256>>>(s.rp.p(),s.ci.p(),s.v.p(),s.flags.p(),s.c.m,s.b.p(),s.x.p(),s.map.p(),s.c.length);CK(cudaGetLastError());}
bool check(State&s,std::vector<double>&oracle){bool ok=true;auto x=s.x.get();auto flags=s.flags.get();for(int i=0;i<s.c.m+16;i++){ok&=x[i]==(i>=G&&i<s.c.m+G?oracle[s.c.oo+i-G]:GUARD);ok&=flags[i]==(i>=G&&i<s.c.m+G?1:int(GUARD));}return ok;}
int main(){auto f=readv("input_f64.bin"),o=readv("oracle_f64.bin");std::vector<State*>states;for(auto c:cases){auto*s=new State(c,f);launch(*s);CK(cudaDeviceSynchronize());printf("WARM m=%d correct=%d\n",c.m,int(check(*s,o)));fflush(stdout);states.push_back(s);}for(int trial=1;trial<=5;trial++){std::vector<double>pre,post;bool ok=true;double sec=0;for(auto*s:states){s->x.reset();s->flags.reset();CK(cudaDeviceSynchronize());capture(*s,pre);for(int r=0;r<R;r++){s->x.reset();s->flags.reset();CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();launch(*s);CK(cudaDeviceSynchronize());sec+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/R;ok&=check(*s,o);append(post,s->x.get());append(post,s->flags.get());}capture(*s,post);ok&=s->rp.get()==s->rp.h&&s->ci.get()==s->ci.h&&s->map.get()==s->map.h&&s->v.get()==s->v.h&&s->b.get()==s->b.h;}pre.insert(pre.end(),post.begin(),post.end());std::string name="sptrsv_trial"+std::to_string(trial)+".bin";std::ofstream(name,std::ios::binary).write((char*)pre.data(),pre.size()*8);printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":3,\"timing_region\":\"sum_five_custom_syncfree_solves_reset_outside_each_interval_completion_per_solve\",\"max_abs_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"}}\n",trial,ok?"true":"false",sec,name.c_str(),pre.size(),pre.size());fflush(stdout);}}
