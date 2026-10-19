#define main original_cc_main
#include "../cc-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include "study_cases.h"
static void ck(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));exit(3);}}
static std::vector<int> load(const std::string&f,int count){std::ifstream s(f,std::ios::binary);std::vector<int>x(count);s.read((char*)x.data(),count*4);if(!s||s.peek()!=EOF)throw std::runtime_error("input size "+f);return x;}
static std::vector<int> host(int*p,int n){std::vector<int>x(n);ck(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost));return x;}
static void cat(std::vector<int>&a,const std::vector<int>&b){a.insert(a.end(),b.begin(),b.end());}
__global__ void observe_params(int n,int e,int blocks,int*p){p[0]=n;p[1]=e;p[2]=blocks;}
struct Work{Case c;int *idx,*adj,*stat,*wl,*params;std::vector<int>A,B;};
static void pipeline(Work&w){auto c=w.c;init<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16);compute1<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16,w.wl+16);compute2<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16,w.wl+16);compute3<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16,w.wl+16);flatten<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16);}
int main(){cudaDeviceProp dp;ck(cudaGetDeviceProperties(&dp,0));int blocks=dp.multiProcessorCount*dp.maxThreadsPerMultiProcessor/256;if(blocks!=640)throw std::runtime_error("source geometry differs from declared fixture");std::vector<Work>w;
for(auto c:cases){Work x;x.c=c;x.A=load("case"+std::to_string(c.id)+"_idx_i32.bin",c.ic);x.B=load("case"+std::to_string(c.id)+"_adj_i32.bin",c.ac);ck(cudaMalloc(&x.idx,c.ic*4));ck(cudaMalloc(&x.adj,c.ac*4));ck(cudaMalloc(&x.stat,(c.n+32)*4));ck(cudaMalloc(&x.wl,(c.n+32)*4));ck(cudaMalloc(&x.params,12));ck(cudaMemcpy(x.idx,x.A.data(),c.ic*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(x.adj,x.B.data(),c.ac*4,cudaMemcpyHostToDevice));std::vector<int>g(c.n+32,GUARD);ck(cudaMemcpy(x.stat,g.data(),g.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(x.wl,g.data(),g.size()*4,cudaMemcpyHostToDevice));w.push_back(x);}
auto expected=load("oracle_i32.bin",COUNT);
for(auto &x:w){for(int k=0;k<x.c.repeat;k++)pipeline(x);ck(cudaDeviceSynchronize());}
for(int trial=1;trial<=5;trial++){std::vector<int>pre,out;double total=0;std::vector<double>times;
for(auto &x:w){auto c=x.c;std::vector<int>g(c.n+32,GUARD);ck(cudaMemcpy(x.stat,g.data(),g.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(x.wl,g.data(),g.size()*4,cudaMemcpyHostToDevice));auto ib=host(x.idx,c.ic),ab=host(x.adj,c.ac);observe_params<<<1,1>>>(c.n,c.e,c.blocks,x.params);ck(cudaDeviceSynchronize());auto params=host(x.params,3);auto t=std::chrono::steady_clock::now();for(int k=0;k<c.repeat;k++)pipeline(x);ck(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/c.repeat;times.push_back(elapsed);total+=elapsed;
cat(pre,{c.n,c.e,c.repeat,c.blocks});cat(pre,params);cat(pre,ib);cat(pre,ab);cat(pre,host(x.idx,c.ic));cat(pre,host(x.adj,c.ac));cat(out,host(x.stat,c.n+32));auto wl=host(x.wl,c.n+32);out.insert(out.end(),wl.begin(),wl.begin()+16);out.insert(out.end(),wl.end()-16,wl.end());}
cat(pre,out);bool ok=pre==expected;std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/cc_native_trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)pre.data(),pre.size()*4);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%s,\"max_rel_error\":%s,\"jit_excluded\":true,\"timing_region\":\"sum3_source5stage_pipeline_batch_means_R100\",\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,total,ok?"true":"false",pre.size(),ok?"0":"null",ok?"0":"null",path.c_str(),pre.size());fflush(stdout);if(!ok)return 4;}
for(auto&x:w){ck(cudaFree(x.idx));ck(cudaFree(x.adj));ck(cudaFree(x.stat));ck(cudaFree(x.wl));ck(cudaFree(x.params));}return 0;}
