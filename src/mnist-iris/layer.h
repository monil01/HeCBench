#pragma once
#include <iris/iris.hpp>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
const static float dt=1.0E-1f;
const static float threshold=1.0E-2f;
extern int gpu_device;
struct Buffer {
  std::vector<float> host;
  std::unique_ptr<iris::DMem> mem;
  explicit Buffer(int n):host(n){}
  void bind(){if(!host.empty()) mem=std::make_unique<iris::DMem>(host.data(),host.size()*sizeof(float));}
  void reset(){std::fill(host.begin(),host.end(),0.f);mem->update(host.data());}
  void fetch(){iris::Task task;task.flush_out(*mem);
    if(task.submit(gpu_device,nullptr,1)) throw std::runtime_error("fetch failed");}
};
struct Layer {
  int M,N,O;
  Buffer output,preact,bias,weight,d_output,d_preact,d_weight;
  Layer(int m,int n,int o):M(m),N(n),O(o),output(o),preact(o),bias(n),weight(m*n),d_output(o),d_preact(o),d_weight(m*n) {
    for(int i=0;i<N;++i) {
      bias.host[i]=0.5f-float(rand())/float(RAND_MAX);
      for(int j=0;j<M;++j) weight.host[i*M+j]=0.5f-float(rand())/float(RAND_MAX);
    }
  }
  void bind(){output.bind();preact.bind();bias.bind();weight.bind();d_output.bind();d_preact.bind();d_weight.bind();}
  void clear(){output.reset();preact.reset();}
  void bp_clear(){d_output.reset();d_preact.reset();d_weight.reset();}
  void setOutput(const float* values){std::copy(values,values+O,output.host.begin());output.mem->update(output.host.data());}
};
struct Param {void* ptr;int info;};
static Param param(Buffer& value){return {value.mem.get(),iris_rw};}
template<class T> static Param param(const T& value){return {const_cast<T*>(&value),sizeof(T)};}
template<class... Args> static void launch(const char* name,size_t blocks,size_t threads,Args&&... args){
  Param list[]={param(args)...};void* p[sizeof...(args)];int info[sizeof...(args)];
  for(size_t i=0;i<sizeof...(args);++i){p[i]=list[i].ptr;info[i]=list[i].info;}
  size_t g[]={blocks*threads},l[]={threads};iris::Task task;
  if(task.kernel(name,1,nullptr,g,l,sizeof...(args),p,info)||task.submit(gpu_device,nullptr,1)) throw std::runtime_error("MNIST task failed");
}
