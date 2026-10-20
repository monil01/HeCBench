#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <map>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cstdint>
namespace study {
inline void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Allocation{void*base;size_t n;int id;};
struct State{std::map<void*,Allocation>live;int next=0;bool capture=false,running=false,pending=false,training=true;int step=0,label=0,oldstep=0,oldlabel=0;double seconds=0;std::chrono::steady_clock::time_point tick;std::vector<double>prefix,output;std::ofstream science,physical;std::string folder;int trial=0;bool correct=true;};
inline State&state(){static State s;return s;}
struct Pause{bool active;Pause(){auto&s=state();active=s.running;if(active){check(cudaDeviceSynchronize());s.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-s.tick).count();s.running=false;}}~Pause(){if(active){auto&s=state();s.tick=std::chrono::steady_clock::now();s.running=true;}}};
inline void write(std::ofstream&f,const void*p,size_t n){if(n){f.write((const char*)p,n);if(!f)throw std::runtime_error("capture write failed");}}
inline void allocate(float**p,size_t n){auto&s=state();void*base=nullptr;check(cudaMalloc(&base,n+512));check(cudaMemset(base,0xa7,n+512));*p=(float*)((char*)base+256);s.live[*p]={base,n,s.next++};}
inline void release(void*p){if(!p)return;auto&s=state();auto it=s.live.find(p);if(it==s.live.end())std::abort();check(cudaFree(it->second.base));s.live.erase(it);}
inline std::vector<float>read(float*p,int n){std::vector<float>x(n);if(n)check(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost));return x;}
void before_upload();void after_upload(float*,int);void forward_capture();void test_capture(unsigned);void snapshot(int,int,int,int);
inline void training_step(int i,int label){auto&s=state();s.training=true;s.step=i;s.label=label;}
inline void testing_step(int i,int label){auto&s=state();s.training=false;s.step=i;s.label=label;}
inline void epoch_error(float e){auto&s=state();if(s.capture){Pause pause;s.output.push_back(30);s.output.push_back(e);s.correct&=std::isfinite(e)&&e>=.01f;}}
inline void test_errors(int errors){auto&s=state();if(s.capture){Pause pause;s.output.push_back(31);s.output.push_back(errors);}}
}
