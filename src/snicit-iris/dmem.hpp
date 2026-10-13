#pragma once
#include <iris/iris.hpp>
#include <hip/hip_runtime.h>
#include <memory>
#include <cstring>
#include <cstdio>
#include <cstdlib>
inline void iris_check(int rc) { if(rc!=IRIS_SUCCESS) { fprintf(stderr,"IRIS error %d\n",rc);exit(1); } }
template<class T> struct SBuffer {
  size_t bytes;
  std::unique_ptr<T[]> host;
  iris::DMem mem;
  explicit SBuffer(size_t n):bytes(n),host(new T[(n+sizeof(T)-1)/sizeof(T)]),mem(host.get(),n) {}
  T &operator[](size_t i) { return host[i]; }
  void update(const void *source,size_t n) { memcpy(host.get(),source,n);update_binding(); }
  void update_binding() { iris_check(mem.update(host.get())); }
  void zero(size_t n) { memset(host.get(),0,n);update_binding(); }
  void flush() { iris::Task task;iris_check(task.flush_out(mem));iris_check(task.submit(iris_gpu,nullptr,true)); }
};
template<class T> using SMemory=std::shared_ptr<SBuffer<T>>;
inline void sn_launch(const char *name,int blocks,dim3 threads,int count,void **args,int *info) {
  if(blocks==0)return;
  size_t gws[]={size_t(blocks)*threads.x,threads.y,threads.z},lws[]={threads.x,threads.y,threads.z};
  iris::Task task;iris_check(task.kernel(name,3,nullptr,gws,lws,count,args,info));iris_check(task.submit(iris_gpu,nullptr,true));
}
