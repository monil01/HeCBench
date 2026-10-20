// Source device kernels unchanged; deterministic supported-domain instrumentation.
#include "original_native/warpsort.cu"
#include "cases.h"
#include <fstream>
#include <algorithm>
#include <map>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <iomanip>
using namespace facebook::cuda;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static float fl(uint32_t x){float f;memcpy(&f,&x,4);return f;}
static uint32_t bits(float x){uint32_t b;memcpy(&b,&x,4);return b;}
static constexpr uint32_t GUARD=0xa5a5a5a5;
struct Buf{uint32_t *p;int n;Buf(int n):n(n){ck(cudaMalloc(&p,(n+64)*4));ck(cudaMemset(p,0xa5,(n+64)*4));}~Buf(){cudaFree(p);}void* data(){return p+32;}std::vector<uint32_t> read(){std::vector<uint32_t> h(n+64);ck(cudaMemcpy(h.data(),p,h.size()*4,cudaMemcpyDeviceToHost));return h;}};
static double run(int trial,bool emit){std::vector<double> prefix,output;double elapsed=0;for(auto &c:CASES){Buf a(c.n),b(c.n),ix(c.n);ck(cudaMemcpy(a.data(),c.bits.data(),c.n*4,cudaMemcpyHostToDevice));ck(cudaMemset(b.data(),0,c.n*4));ck(cudaMemset(ix.data(),0,c.n*4));auto ai=a.read();prefix.push_back(c.n);prefix.push_back(c.mode);for(int i=0;i<c.n;i++){if(ai[i+32]!=c.bits[i])throw std::runtime_error("input");prefix.push_back(ai[i+32]);}int sz[]={c.n};auto da=DeviceTensor<float,1>((float*)a.data(),sz),db=DeviceTensor<float,1>((float*)b.data(),sz);auto di=DeviceTensor<int,1>((int*)ix.data(),sz);ck(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();if(c.mode)sortDevice<<<1,32>>>(da,db,di);else sortDevice<<<1,32>>>(da,db);ck(cudaGetLastError());ck(cudaDeviceSynchronize());elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();auto ah=a.read(),bh=b.read(),ih=ix.read();std::map<uint32_t,int> count;for(auto x:c.bits)count[x]++;for(int i=0;i<c.n;i++){float v=fl(bh[i+32]);if(i&&fl(bh[i+31])<v)throw std::runtime_error("order");count[bh[i+32]]--;output.push_back(bh[i+32]);}for(auto kv:count)if(kv.second)throw std::runtime_error("multiset");if(c.mode){std::vector<int> want(c.n);for(int i=0;i<c.n;i++)want[i]=i;std::sort(want.begin(),want.end(),[&](int i,int j){return fl(c.bits[i])>fl(c.bits[j])||(fl(c.bits[i])==fl(c.bits[j])&&i>j);});for(int i=0;i<c.n;i++){if((int)ih[i+32]!=want[i]||bh[i+32]!=c.bits[want[i]])throw std::runtime_error("index");output.push_back((int)ih[i+32]);}}for(auto &h:{ah,bh,ih})for(int side=0;side<2;side++)for(int j=0;j<32;j++){auto x=h[side?(c.n+32+j):j];output.push_back(x);if(x!=GUARD)throw std::runtime_error("guard");}}
if(emit){prefix.insert(prefix.end(),output.begin(),output.end());std::string name="trial"+std::to_string(trial)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)prefix.data(),prefix.size()*8);f.close();if(!f)throw std::runtime_error("write");std::cout<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<std::setprecision(17)<<elapsed<<",\"correct\":true,\"checked_values\":"<<prefix.size()<<",\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"sum258_source_singlewarp_kernel_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\""<<name<<"\",\"dtype\":\"float64\",\"count\":"<<prefix.size()<<",\"byte_order\":\"little\"}}\n";}
return elapsed;}
int main(){try{run(0,false);for(int i=1;i<=5;i++)run(i,true);return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
