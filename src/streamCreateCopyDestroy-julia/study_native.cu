// Derived runtime call ordering from ../streamCreateCopyDestroy-cuda/main.cu.
// Original AMD MIT copyright/license retained in LICENSE.
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <array>
using U=unsigned;constexpr int N=4096,G=64;constexpr float SENT=-777.f;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
U raw(float x){U u;memcpy(&u,&x,4);return u;}
struct Case{int test,mode,s,b,it;std::vector<float> h;std::vector<float*> d;std::vector<cudaStream_t> streams;};
std::vector<U> oracle;
void reset(Case&c){for(auto d:c.d)ck(cudaMemset(d+G,255,N*4));ck(cudaDeviceSynchronize());}
std::array<U,4> pipeline(Case&c){std::array<U,4> calls{};for(int i=0;i<c.it;i++){if(c.mode){for(auto&s:c.streams){ck(cudaStreamCreate(&s));calls[1]++;}for(auto s:c.streams)for(auto d:c.d){ck(cudaMemcpyAsync(d+G,c.h.data()+G,N*4,cudaMemcpyHostToDevice,s));calls[0]++;}for(auto s:c.streams){ck(cudaStreamSynchronize(s));calls[3]++;ck(cudaStreamDestroy(s));calls[2]++;}}else{for(int s=0;s<c.s;s++)for(auto d:c.d){ck(cudaMemcpyAsync(d+G,c.h.data()+G,N*4,cudaMemcpyHostToDevice));calls[0]++;}ck(cudaDeviceSynchronize());calls[3]++;}}return calls;}
int main(){try{std::ifstream f(std::string(CCD_ROOT)+"/oracle_u32.bin",std::ios::binary);oracle.resize(N);f.read((char*)oracle.data(),N*4);if(!f||f.peek()!=EOF)throw std::runtime_error("oracle read");std::vector<Case> cases;for(int mode=0;mode<2;mode++)for(int test:std::vector<int>{0,5,10,15}){int group=test/4;int ss[]={1,2,4,8},bb[]={1,100,1000,5000};Case c{test,mode,ss[test%4],bb[group],256/(ss[test%4]*(1<<(group+1))),std::vector<float>(N+2*G,SENT),{}};for(int i=0;i<N;i++){c.h[G+i]=1.618f+i;if(raw(c.h[G+i])!=oracle[i])throw std::runtime_error("input oracle");}for(int b=0;b<c.b;b++){float*d;ck(cudaMalloc(&d,(N+2*G)*4));ck(cudaMemcpy(d,c.h.data(),(N+2*G)*4,cudaMemcpyHostToDevice));c.d.push_back(d);}c.streams.resize(c.s);cases.push_back(std::move(c));}
for(auto&c:cases){reset(c);pipeline(c);}for(int trial=1;trial<=5;trial++){std::vector<U> out;out.reserve(51574089);out.push_back(8);for(auto&c:cases){out.insert(out.end(),{U(c.test),U(c.mode),U(c.s),U(c.b),U(c.it)});for(int i=0;i<N;i++)out.push_back(raw(c.h[G+i]));}double total=0;std::vector<double>times;bool ok=true;for(auto&c:cases){reset(c);auto a=std::chrono::steady_clock::now();auto calls=pipeline(c);auto z=std::chrono::steady_clock::now();double dt=std::chrono::duration<double>(z-a).count();total+=dt;times.push_back(dt);std::vector<float>h(N+2*G);for(auto d:c.d){ck(cudaMemcpy(h.data(),d,h.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<N+2*G;i++){U v=raw(h[i]);ok &=v==((i<G||i>=G+N)?raw(SENT):oracle[i-G]);out.push_back(v);}}for(int i=0;i<N;i++)ok &=raw(c.h[G+i])==oracle[i];for(int i=0;i<G;i++)ok &=c.h[i]==SENT&&c.h[G+N+i]==SENT;out.insert(out.end(),calls.begin(),calls.end());}std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream o(path,std::ios::binary);o.write((char*)out.data(),out.size()*4);if(!o)throw std::runtime_error("capture write");std::cout.precision(17);std::cout<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<total<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":"<<(ok?0:1)<<",\"max_rel_error\":null,\"timing_region\":\"sum8_pageable_H2D_stream_create_sync_destroy_runtime_pipeline\",\"jit_excluded\":true,\"internal_repeats\":1,\"authored_device_kernel_calls\":0,\"case_times_seconds\":[";for(int i=0;i<8;i++)std::cout<<(i?",":"")<<times[i];std::cout<<"],\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"uint32\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}\n";if(!ok)throw std::runtime_error("copy input output guards mismatch");}for(auto&c:cases)for(auto d:c.d)ck(cudaFree(d));return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
