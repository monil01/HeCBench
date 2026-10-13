#pragma once
#include <iris/iris.hpp>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <initializer_list>
struct dim3 { size_t x,y,z; dim3(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c){} };
struct Buffer {
  std::vector<unsigned char> host;
  iris::DMem mem;
  explicit Buffer(size_t n):host(n,0),mem(host.data(),n){}
};
inline std::vector<std::unique_ptr<Buffer>>& buffers(){static std::vector<std::unique_ptr<Buffer>> b;return b;}
inline Buffer& findBuffer(const void* p) {
  auto q=reinterpret_cast<uintptr_t>(p);
  for(auto& b:buffers()) {auto base=reinterpret_cast<uintptr_t>(b->host.data());if(q>=base&&q<base+b->host.size()) return *b;}
  throw std::runtime_error("Unknown dmem view");
}
inline void flush(Buffer& b) {
  iris::Task task;task.flush_out(b.mem);
  if(task.submit(iris_gpu,nullptr,true)!=IRIS_SUCCESS)throw std::runtime_error("flush");
}
inline void allocateBuffer(void** p,size_t n){auto b=std::make_unique<Buffer>(n);*p=b->host.data();buffers().push_back(std::move(b));}
inline void releaseBuffer(void* p){auto& bs=buffers();for(auto it=bs.begin();it!=bs.end();++it)if((*it)->host.data()==p){bs.erase(it);return;}throw std::runtime_error("free view");}
inline void uploadBuffer(void* p,const void* src,size_t n){auto& b=findBuffer(p);flush(b);std::memcpy(p,src,n);if(b.mem.update(b.host.data())!=IRIS_SUCCESS)throw std::runtime_error("update");}
inline void resetBuffer(void* p,int v,size_t n){auto& b=findBuffer(p);flush(b);std::memset(p,v,n);if(b.mem.update(b.host.data())!=IRIS_SUCCESS)throw std::runtime_error("reset update");}
inline void downloadBuffer(void* out,const void* p,size_t n){auto& b=findBuffer(p);flush(b);std::memcpy(out,p,n);}
struct Argument {Buffer* b=nullptr;size_t offset=0;std::vector<unsigned char> bytes;};
inline Argument memory(const void* p){auto& b=findBuffer(p);return {&b,size_t(static_cast<const unsigned char*>(p)-b.host.data()),{}};}
template<class T> Argument scalar(T v){Argument a;a.bytes.resize(sizeof(T));std::memcpy(a.bytes.data(),&v,sizeof(T));return a;}
inline void submitKernel(const char* name,size_t blocks,size_t threads,std::initializer_list<Argument> args){
  std::vector<void*> params;std::vector<int> info;
  for(auto& a:args){params.push_back(a.b?static_cast<void*>(&a.b->mem):const_cast<unsigned char*>(a.bytes.data()));info.push_back(a.b?iris_rw:int(a.bytes.size()));}
  // All dmem pointers are bases; GPU wrapper applies each distinct byte offset.
  for(auto& a:args)if(a.b){params.push_back(const_cast<size_t*>(&a.offset));info.push_back(sizeof(size_t));}
  size_t gws[]={blocks*threads},lws[]={threads};iris::Task task;
  if(task.kernel(name,1,nullptr,gws,lws,params.size(),params.data(),info.data())!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,true)!=IRIS_SUCCESS)throw std::runtime_error(name);
}
inline int require_amd_hip(){
  int n=0;iris_device_count(&n);
  for(int i=0;i<n;++i){int backend=0,type=0;char name[256]={};iris_device_info(i,iris_backend,&backend,nullptr);iris_device_info(i,iris_type,&type,nullptr);iris_device_info(i,iris_name,name,nullptr);std::printf("IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);if(backend==3&&type==256)return i;}
  return -1;
}
