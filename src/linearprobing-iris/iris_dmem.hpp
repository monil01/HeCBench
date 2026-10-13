#pragma once
#include <iris/iris.hpp>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
using uint = unsigned int;
struct dim3 {
  size_t x,y,z;
  dim3(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c){}
};
inline void iris_require(int code) {
  if(code != IRIS_SUCCESS) { fprintf(stderr,"IRIS failure %d\n",code); std::exit(1); }
}
struct IRISBuffer {
  void* host; size_t bytes; iris::DMem mem;
  explicit IRISBuffer(size_t n):host(std::calloc(n,1)),bytes(n),mem(host,n) {
    if(!host) { fprintf(stderr,"Host allocation failed\n"); std::exit(1); }
  }
  ~IRISBuffer() { std::free(host); }
};
inline std::map<void*,std::unique_ptr<IRISBuffer>> iris_buffers;
struct IRISSession {
  iris::Platform platform;
  IRISSession(int* argc,char*** argv) {
    iris_require(platform.init(argc,argv,1));
    int count=0; iris_require(iris_device_count(&count));
    bool hip=false;
    for(int i=0;i<count;++i) {
      int backend=0; char name[256]={};
      iris_require(iris_device_info(i,iris_backend,&backend,nullptr));
      iris_require(iris_device_info(i,iris_name,name,nullptr));
      printf("IRIS device %d: %s backend=%d\n",i,name,backend);
      hip |= backend==iris_hip;
      if(backend!=iris_hip) { fprintf(stderr,"Require IRIS_ARCHS=HIP\n"); std::exit(1); }
    }
    if(!hip) { fprintf(stderr,"No HIP GPU\n"); std::exit(1); }
  }
  ~IRISSession() { iris_buffers.clear(); iris_require(platform.finalize()); }
};
template<class P> int iris_alloc(P* p,size_t bytes) {
  auto b=std::make_unique<IRISBuffer>(bytes); *p=reinterpret_cast<P>(b->host);
  iris_buffers.emplace(b->host,std::move(b)); return 0;
}
inline int iris_alloc(void** p,size_t bytes) { return iris_alloc<void*>(p,bytes); }
inline IRISBuffer& iris_buffer(const void* p) {
  auto it=iris_buffers.find(const_cast<void*>(p));
  if(it==iris_buffers.end()) { fprintf(stderr,"Unknown DMem pointer\n"); std::exit(1); }
  return *it->second;
}
enum IRISCopy { IRISHostToDevice, IRISDeviceToHost, IRISDeviceToDevice };
inline int iris_copy(void* dst,const void* src,size_t bytes,IRISCopy kind) {
  if(kind==IRISHostToDevice) {
    auto& b=iris_buffer(dst); if(bytes>b.bytes) std::exit(1);
    std::memcpy(b.host,src,bytes); iris_require(b.mem.update(b.host));
  } else if(kind==IRISDeviceToHost) {
    auto& b=iris_buffer(src); if(bytes>b.bytes) std::exit(1);
    iris::Task t; iris_require(t.flush_out(b.mem)); iris_require(t.submit(iris_gpu,nullptr,1));
    std::memcpy(dst,b.host,bytes);
  } else {
    auto& s=iris_buffer(src); auto& d=iris_buffer(dst);
    iris::Task t; iris_require(t.flush_out(s.mem)); iris_require(t.submit(iris_gpu,nullptr,1));
    std::memcpy(d.host,s.host,bytes); iris_require(d.mem.update(d.host));
  }
  return 0;
}
inline int iris_set(void* dst,int value,size_t bytes) {
  auto& b=iris_buffer(dst); if(bytes>b.bytes) std::exit(1);
  // Preserve untouched bytes for partial memset.
  if(bytes<b.bytes) { iris::Task t; iris_require(t.flush_out(b.mem)); iris_require(t.submit(iris_gpu,nullptr,1)); }
  std::memset(b.host,value,bytes); iris_require(b.mem.update(b.host)); return 0;
}
inline int iris_free(void* p) { iris_buffers.erase(p); return 0; }
inline int iris_sync() { return 0; } // Each task submits synchronously.
inline const char* iris_error_string(int) { return "IRIS task or data memory error"; }
template<class T> void iris_param(T& value,void*& ptr,int& info) {
  if constexpr(std::is_pointer_v<T>) { ptr=&iris_buffer(value).mem; info=iris_rw; }
  else { ptr=&value; info=sizeof(T); }
}
template<class... Args> void iris_launch(const char* name,dim3 grid,dim3 block,Args... args) {
  auto values=std::make_tuple(args...);
  std::array<void*,sizeof...(Args)> params{};
  std::array<int,sizeof...(Args)> info{};
  size_t i=0;
  // Index explicitly so parameter order is identical to the CUDA signature.
  i=0; std::apply([&](auto&... x){ ((iris_param(x,params[i],info[i]),++i),...); },values);
  size_t gws[]={grid.x*block.x,grid.y*block.y,grid.z*block.z};
  size_t lws[]={block.x,block.y,block.z};
  int dimensions=grid.z*block.z>1?3:(grid.y*block.y>1?2:1);
  iris::Task task;
  iris_require(task.kernel(name,dimensions,nullptr,gws,lws,sizeof...(Args),params.data(),info.data()));
  iris_require(task.submit(iris_gpu,nullptr,1));
}
