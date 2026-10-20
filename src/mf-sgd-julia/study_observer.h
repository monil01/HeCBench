#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <cstdio>
namespace mfobs {
inline void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Allocation {int id;void*base;void*data;size_t bytes;};
struct State {
 bool active=false,capture=false,roi=false,valid=true;int paused=0,case_id=0,next_id=0,workers=0,epochs=30,nnz=0,m=0,n=0;double mark=0,seconds=0;std::map<void*,Allocation>live;std::map<int,std::vector<unsigned char>>released;std::vector<double>prefix;std::vector<int>launches=std::vector<int>(4,0);std::ofstream raw,guards,layout;
 void pause(){if(roi&&paused++==0)seconds+=now()-mark;}
 void resume(){if(roi&&--paused==0)mark=now();}
 void start(){if(roi||paused)throw std::runtime_error("timerstate");roi=true;mark=now();}
 void stop(){check(cudaDeviceSynchronize());if(paused)throw std::runtime_error("timerpause");seconds+=now()-mark;roi=false;}
 void open_case(const std::string&base,int id,int w,int rows,int cols,int nz){if(!live.empty())throw std::runtime_error("liveallocation");active=true;case_id=id;workers=w;m=rows;n=cols;nnz=nz;next_id=0;released.clear();std::fill(launches.begin(),launches.end(),0);if(capture){raw.open(base+".raw.bin",std::ios::binary);guards.open(base+".guards.u8",std::ios::binary);layout.open(base+".layout.jsonl");}}
 void close_case(){if(!live.empty())throw std::runtime_error("liveallocationend");if(capture){raw.close();guards.close();layout.close();}active=false;}
 std::vector<unsigned char>snapshot(const Allocation&a,const char*event){std::vector<unsigned char>b(a.bytes+512);check(cudaMemcpy(b.data(),a.base,b.size(),cudaMemcpyDeviceToHost));for(int i=0;i<256;i++)valid&=b[i]==165&&b[256+a.bytes+i]==165;if(capture){size_t off=size_t(raw.tellp()),go=size_t(guards.tellp());raw.write((char*)b.data(),b.size());guards.write((char*)b.data(),256);guards.write((char*)b.data()+256+a.bytes,256);layout<<"{\"case\":"<<case_id<<",\"id\":"<<a.id<<",\"event\":\""<<event<<"\",\"payload_bytes\":"<<a.bytes<<",\"raw_offset_bytes\":"<<off<<",\"guard_offset_bytes\":"<<go<<",\"guard_bytes\":512}\n";}return std::vector<unsigned char>(b.begin()+256,b.begin()+256+a.bytes);}
 template<class T>void append_typed(const std::vector<unsigned char>&b,size_t count){if(count*sizeof(T)>b.size())throw std::runtime_error("typedpayload");for(size_t i=0;i<count;i++){T x;std::memcpy(&x,b.data()+i*sizeof(T),sizeof(T));prefix.push_back(double(x));}}
};
inline State*state=nullptr;
struct Pause {State*s;Pause(State*p):s(p){if(s)s->pause();}~Pause(){if(s)s->resume();}};
inline cudaError_t allocate(void**data,size_t bytes){if(!state||!state->active)return cudaMalloc(data,bytes);void*base=nullptr;cudaError_t e=cudaMalloc(&base,bytes+512);if(e!=cudaSuccess)return e;check(cudaDeviceSynchronize());{Pause pause(state);check(cudaMemset(base,165,256));check(cudaMemset((unsigned char*)base+256+bytes,165,256));check(cudaDeviceSynchronize());}*data=(unsigned char*)base+256;state->live[*data]=Allocation{state->next_id++,base,*data,bytes};return cudaSuccess;}
inline cudaError_t release(void*data){if(!state)return cudaFree(data);auto it=state->live.find(data);if(it==state->live.end())return cudaFree(data);Allocation a=it->second;check(cudaDeviceSynchronize());{Pause pause(state);state->released[a.id]=state->snapshot(a,"release");}cudaError_t e=cudaFree(a.base);state->live.erase(it);return e;}
inline void launch(int code){if(state&&state->active)state->launches.at(code)++;}
inline void before_feature(int state_size,long long array_size,long long k,float scale,int grid){if(!state||!state->active)return;check(cudaDeviceSynchronize());Pause pause(state);state->prefix.insert(state->prefix.end(),{double(state->case_id),double(state_size),double(array_size),double(k),double(scale),double(grid)});auto it=state->live.end();for(auto q=state->live.begin();q!=state->live.end();q++)if(q->second.id==1)it=q;if(it==state->live.end())throw std::runtime_error("featurestateallocation");auto b=state->snapshot(it->second,"actual_feature_state_input");state->append_typed<uint32_t>(b,state_size);}
inline void before_training(){if(!state||!state->active)return;check(cudaDeviceSynchronize());Pause pause(state);state->prefix.insert(state->prefix.end(),{double(state->case_id),double(state->workers),double(state->epochs),double(state->m),double(state->n),double(state->nnz),128.0});for(int id=0;id<5;id++){auto found=state->live.end();for(auto it=state->live.begin();it!=state->live.end();it++)if(it->second.id==id)found=it;if(found==state->live.end())throw std::runtime_error("inputallocationmissing");auto b=state->snapshot(found->second,"actual_before_training");state->prefix.insert(state->prefix.end(),{double(id),double(b.size())});if(id==0)state->append_typed<uint32_t>(b,state->workers);else if(id==1)state->append_typed<float>(b,state->epochs+4);else if(id==2){if(b.size()!=size_t(state->nnz)*12)throw std::runtime_error("nodelayout");for(int i=0;i<state->nnz;i++){int32_t u,v;float rate;std::memcpy(&u,b.data()+12*i,4);std::memcpy(&v,b.data()+12*i+4,4);std::memcpy(&rate,b.data()+12*i+8,4);state->prefix.insert(state->prefix.end(),{double(u),double(v),double(rate)});}}else state->append_typed<uint16_t>(b,b.size()/2);}}
}
inline long long*study_trace_ptr=nullptr;
