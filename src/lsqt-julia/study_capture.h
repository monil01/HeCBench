#pragma once
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <unistd.h>
namespace lsqtstudy {
using Clock=std::chrono::steady_clock;
struct Allocation{void*base;void*data;size_t bytes;int id;bool integer;std::string name;};
struct State{
 bool capture=false,roi=false,correct=true;int paused=0,case_id=0,trial=0,r=0,kind=0,step=0,orbital=-1,next_id=0;
 Clock::time_point mark;double seconds=0;std::string folder,base;std::map<void*,Allocation>live;std::map<int,std::vector<double>>released;
 std::vector<double>prefix,output;std::ofstream raw,guards,layout,events;std::array<size_t,14>launches{},timed_launches{};
};
inline State& state(){static State s;return s;}
inline void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline void bytes(std::ofstream&f,const void*p,size_t n){f.write((const char*)p,n);if(!f)throw std::runtime_error("capture write failed");}
inline void pause(){auto&s=state();if(s.roi&&s.paused++==0)s.seconds+=std::chrono::duration<double>(Clock::now()-s.mark).count();}
inline void resume(){auto&s=state();if(s.roi&&--s.paused==0)s.mark=Clock::now();}
struct Pause{Pause(){pause();}~Pause(){resume();}};
inline void roi_begin(int kind){auto&s=state();check(cudaDeviceSynchronize());if(s.roi)throw std::runtime_error("nested ROI");s.kind=kind;s.step=0;s.paused=0;s.roi=true;s.mark=Clock::now();}
inline void roi_end(){auto&s=state();check(cudaDeviceSynchronize());if(!s.roi||s.paused)throw std::runtime_error("ROI state");s.seconds+=std::chrono::duration<double>(Clock::now()-s.mark).count();s.roi=false;}
inline void launch(int code){auto&s=state();s.launches.at(code)++;if(s.roi)s.timed_launches.at(code)++;}
inline void set_step(int step){state().step=step;}
inline void scalar(int code,std::initializer_list<double>v){auto&s=state();if(!s.capture)return;/* Included observer overhead: do not subtract hosttime overlapping activeGPUwork. */s.prefix.insert(s.prefix.end(),{double(code),double(s.case_id),double(s.kind),double(s.r),double(s.step),double(s.orbital),double(v.size())});s.prefix.insert(s.prefix.end(),v.begin(),v.end());}
inline cudaError_t allocate(void**pointer,size_t n,const char*expression){
 auto&s=state(); // Original malloc cost stays in the timer; enlarged guards disclosed.
 void*base=nullptr;check(cudaMalloc(&base,n+512));check(cudaDeviceSynchronize());*pointer=(unsigned char*)base+256;
 {Pause p;check(cudaMemset(base,165,n+512));check(cudaDeviceSynchronize());}
 bool integer=std::string(expression).find("neighbor_")!=std::string::npos;
 s.live.emplace(*pointer,Allocation{base,*pointer,n,s.next_id++,integer,expression});return cudaSuccess;
}
inline std::vector<unsigned char> physical(const Allocation&a){std::vector<unsigned char>b(a.bytes+512);check(cudaMemcpy(b.data(),a.base,b.size(),cudaMemcpyDeviceToHost));return b;}
inline void guard(const Allocation&a,const std::vector<unsigned char>&b){
 auto&s=state();for(size_t i=0;i<256;i++)s.correct&=b[i]==165&&b[256+a.bytes+i]==165;
 if(s.capture){bytes(s.guards,b.data(),256);bytes(s.guards,b.data()+256+a.bytes,256);}
}
inline std::vector<double> values(const Allocation&a,const std::vector<unsigned char>&b){
 std::vector<double>v;size_t width=a.integer?4:8;if(a.bytes%width)throw std::runtime_error("typed allocation width");v.reserve(a.bytes/width);
 for(size_t i=0;i<a.bytes;i+=width){double x;if(a.integer){int t;memcpy(&t,b.data()+256+i,4);x=t;}else memcpy(&x,b.data()+256+i,8);state().correct&=std::isfinite(x);v.push_back(x);}return v;
}
inline std::vector<double> snapshot(void*pointer,const char*event){
 auto&s=state();const auto&a=s.live.at(pointer);auto b=physical(a);
 if(s.capture){s.layout<<"{\"kind\":\"gpu_buffer\",\"event\":\""<<event<<"\",\"id\":"<<a.id<<",\"dtype\":\""<<(a.integer?"int32":"float64")<<"\",\"payload_bytes\":"<<a.bytes<<",\"raw_offset_bytes\":"<<s.raw.tellp()<<",\"payload_offset_bytes\":256,\"guard_offset_bytes\":"<<s.guards.tellp()<<",\"guard_bytes\":512,\"stage\":"<<s.kind<<",\"random\":"<<s.r<<",\"step\":"<<s.step<<",\"orbital\":"<<s.orbital<<"}\n";bytes(s.raw,b.data(),b.size());}
 guard(a,b);return values(a,b);
}
inline cudaError_t release(void*pointer){
 auto&s=state();auto a=s.live.at(pointer);{Pause p;if(s.capture)s.released.emplace(a.id,snapshot(pointer,"release"));else{auto b=physical(a);guard(a,b);}}
 check(cudaFree(a.base));s.live.erase(pointer);return cudaSuccess;
}
inline void tag(int code,int count){auto&s=state();s.output.insert(s.output.end(),{double(code),double(s.case_id),double(s.kind),double(s.r),double(s.step),double(s.orbital),double(count)});}
inline void moments(int n,const double*re,const double*im){auto&s=state();if(!s.capture)return;Pause p;tag(1,n);s.output.insert(s.output.end(),re,re+n);s.output.insert(s.output.end(),im,im+n);}
inline void scientific(int n,const double*out){auto&s=state();if(!s.capture)return;Pause p;tag(2,n);s.output.insert(s.output.end(),out,out+n);}
inline void final_state(int role,int n,void*re,void*im){auto&s=state();if(!s.capture)return;Pause p;tag(3,n);s.output.push_back(role);auto a=snapshot(re,"final_state"),b=snapshot(im,"final_state");if(a.size()!=size_t(n)||b.size()!=size_t(n))throw std::runtime_error("state length");s.output.insert(s.output.end(),a.begin(),a.end());s.output.insert(s.output.end(),b.begin(),b.end());}
inline void initial_state(int n,void*re,void*im){
 auto&s=state();if(!s.capture)return;check(cudaDeviceSynchronize());Pause p;
 s.prefix.insert(s.prefix.end(),{double(s.r),double(s.orbital),double(n)});auto a=snapshot(re,"input_state"),b=snapshot(im,"input_state");s.prefix.insert(s.prefix.end(),a.begin(),a.end());s.prefix.insert(s.prefix.end(),b.begin(),b.end());
}
inline void begin_case(std::string folder,std::string model,int trial,int case_id){
 auto&s=state();if(!s.live.empty())throw std::runtime_error("surviving GPU allocation");s.trial=trial;s.case_id=case_id;s.next_id=0;s.r=0;s.kind=0;s.step=0;s.orbital=-1;s.launches.fill(0);s.timed_launches.fill(0);s.released.clear();if(!s.capture)return;
 s.base=folder+"/"+model+"_"+std::to_string(getpid())+"_trial"+std::to_string(trial)+"_case"+std::to_string(case_id);s.raw.open(s.base+".raw.bin",std::ios::binary);s.guards.open(s.base+".guards.u8",std::ios::binary);s.layout.open(s.base+".layout.jsonl");s.events.open(s.base+".work.json");if(!s.raw||!s.guards||!s.layout||!s.events)throw std::runtime_error("persistent capture path");
}
inline void end_case(){auto&s=state();if(!s.live.empty())throw std::runtime_error("GPU lifetime mismatch");if(!s.capture)return;
 // Canonical allocation identity, not physical pointer-address equality.
 for(const auto&item:s.released){s.output.insert(s.output.end(),{4.,double(s.case_id),double(item.first),double(item.second.size())});s.output.insert(s.output.end(),item.second.begin(),item.second.end());}
 s.output.insert(s.output.end(),{5.,double(s.case_id),14.});for(auto n:s.launches)s.output.push_back(n);for(auto n:s.timed_launches)s.output.push_back(n);
 s.events<<"{\"case_id\":"<<s.case_id<<",\"launches\":[";for(int i=0;i<14;i++){if(i)s.events<<',';s.events<<s.launches[i];}s.events<<"],\"timed_launches\":[";for(int i=0;i<14;i++){if(i)s.events<<',';s.events<<s.timed_launches[i];}s.events<<"],\"GPU_allocations\":"<<s.next_id<<"}\n";
 s.raw.close();s.guards.close();s.layout.close();s.events.close();}
}
#define cudaMalloc(pointer,n) lsqtstudy::allocate((void**)(pointer),(n),#pointer)
#define cudaFree(pointer) lsqtstudy::release((void*)(pointer))
