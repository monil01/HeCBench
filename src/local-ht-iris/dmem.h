#pragma once
#include <iris/iris.hpp>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <initializer_list>
#include <vector>


struct DataBuffer {
  std::vector<unsigned char> host;
  iris::DMem mem;
  explicit DataBuffer(size_t n): host(n, 0), mem(host.data(), n) {}
};
struct Argument {
  iris::DMem* mem = nullptr;
  std::vector<unsigned char> bytes;
};
inline Argument memory(void* p) { return {&static_cast<DataBuffer*>(p)->mem, {}}; }
template<class T> Argument scalar(T v) {
  Argument a; a.bytes.resize(sizeof(T)); std::memcpy(a.bytes.data(), &v, sizeof(T)); return a;
}
inline void launch(const char* name, size_t blocks, size_t threads, std::initializer_list<Argument> args) {
  std::vector<void*> params;
  std::vector<int> info;
  for (auto& a: args) {
    params.push_back(a.mem ? static_cast<void*>(a.mem) : const_cast<unsigned char*>(a.bytes.data()));
    // All partial output writes preserve initialized and previously written values.
    info.push_back(a.mem ? iris_rw : static_cast<int>(a.bytes.size()));
  }
  size_t gws[] = {blocks*threads}, lws[] = {threads};
  iris::Task task;
  if (task.kernel(name, 1, nullptr, gws, lws, params.size(), params.data(), info.data()) != IRIS_SUCCESS ||
      task.submit(iris_gpu, nullptr, true) != IRIS_SUCCESS) throw std::runtime_error(name);
}
inline void allocDeviceBuffer(void** p, unsigned long n) { *p = new DataBuffer(n); }
inline void freeDeviceBuffer(void* p) { delete static_cast<DataBuffer*>(p); }
inline void copyToDevice(void* p, void* host, unsigned long n) {
  auto& b=*static_cast<DataBuffer*>(p);
  if (n>b.host.size()) throw std::runtime_error("copy size");
  std::memcpy(b.host.data(),host,n); b.mem.update(b.host.data());
}
inline void copyFromDevice(void* host, void* p, unsigned long n) {
  auto& b=*static_cast<DataBuffer*>(p);
  iris::Task task; task.flush_out(b.mem);
  if(task.submit(iris_gpu,nullptr,true)!=IRIS_SUCCESS) throw std::runtime_error("flush");
  if (n>b.host.size()) throw std::runtime_error("copy size");
  std::memcpy(host,b.host.data(),n);
}
inline void zeroBuffer(void* p, size_t n) {
  auto& b=*static_cast<DataBuffer*>(p);
  if(n>b.host.size()) throw std::runtime_error("zero size");
  std::memset(b.host.data(),0,n); b.mem.update(b.host.data());
}
inline int require_amd_hip() {
  int n=0; iris_device_count(&n);
  for(int i=0;i<n;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    std::printf("IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) return i;
  }
  std::fprintf(stderr,"No HIP AMD device\n"); return -1;
}
