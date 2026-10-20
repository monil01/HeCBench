#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
namespace study{
inline void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Allocation{void*base;size_t n;int id;};
struct State{bool capture=false,running=false,valid=true;int next=0,case_id=0;double seconds=0,max_abs=0,max_fraction=0;uint64_t checked=0;std::chrono::steady_clock::time_point mark;std::map<const void*,Allocation>live;std::vector<float>prefix,output;std::ofstream raw,layout;};
inline State&state(){static State s;return s;}
struct Pause{bool active;Pause(){auto&s=state();active=s.running;if(active){check(cudaDeviceSynchronize());s.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-s.mark).count();s.running=false;}}~Pause(){if(active){auto&s=state();s.mark=std::chrono::steady_clock::now();s.running=true;}}};
inline void bytes(std::ofstream&f,const void*p,size_t n){if(n)f.write((const char*)p,n);if(!f)throw std::runtime_error("write failed");}
template<class T>cudaError_t allocate(T**p,size_t n){auto&s=state();void*base=nullptr;check(cudaMalloc(&base,n+512));{Pause pause;check(cudaMemset(base,0xa7,n+512));check(cudaDeviceSynchronize());}*p=reinterpret_cast<T*>((char*)base+256);s.live[*p]={base,n,s.next++};return cudaSuccess;}
inline cudaError_t release(void*p){auto&s=state();auto a=s.live.at(p);check(cudaDeviceSynchronize());{Pause pause;if(s.capture){std::vector<unsigned char>v(a.n+512);check(cudaMemcpy(v.data(),a.base,v.size(),cudaMemcpyDeviceToHost));for(int i=0;i<256;i++)s.valid&=v[i]==0xa7&&v[a.n+256+i]==0xa7;uint64_t h[]={uint64_t(s.case_id),uint64_t(a.id),uint64_t(a.n)};uint64_t offset=uint64_t(s.raw.tellp());bytes(s.raw,h,24);bytes(s.raw,v.data(),v.size());s.layout<<"{\"case\":"<<s.case_id<<",\"id\":"<<a.id<<",\"payload_bytes\":"<<a.n<<",\"offset_bytes\":"<<offset<<",\"actual_guard_bytes\":512}\n";}}check(cudaFree(a.base));s.live.erase(p);return cudaSuccess;}
inline std::vector<float>read(const float*p,size_t n){std::vector<float>v(n);if(n)check(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}
inline void input_capture(const float*a,const float*b,int w,int h,int stride,float alpha,int levels,int warps,int solvers){auto&s=state();if(!s.capture)return;Pause pause;s.prefix.insert(s.prefix.end(),{float(s.case_id),float(w),float(h),float(stride),alpha,float(levels),float(warps),float(solvers)});for(auto*p:{a,b}){auto v=read(p,size_t(stride)*h);s.prefix.insert(s.prefix.end(),v.begin(),v.end());}}
inline void append_active(const float*p,int w,int h,int stride){auto&s=state();auto v=read(p,size_t(stride)*h);for(int y=0;y<h;y++)s.output.insert(s.output.end(),v.begin()+y*stride,v.begin()+y*stride+w);}
inline void compare(const std::vector<float>&a,const std::vector<double>&r,const std::vector<double>&bounds){auto&s=state();if(a.size()!=r.size()||a.size()!=bounds.size())throw std::runtime_error("oracle size mismatch");for(size_t i=0;i<a.size();i++){double e=std::abs(double(a[i])-r[i]);s.valid&=std::isfinite(a[i])&&e<=bounds[i];s.max_abs=std::max(s.max_abs,e);if(bounds[i]>0)s.max_fraction=std::max(s.max_fraction,e/bounds[i]);s.checked++;}}
}
