#pragma once
#include <iris/iris.hpp>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdio>
#include <cstdlib>
inline void iris_check(int rc) { if(rc!=IRIS_SUCCESS) { fprintf(stderr,"IRIS error %d\n",rc);exit(1); } }
struct LBuffer {
  std::vector<unsigned char> host;
  iris::DMem mem;
  explicit LBuffer(size_t bytes):host(bytes),mem(host.data(),bytes) {}
  void update(const void *source) { memcpy(host.data(),source,host.size());iris_check(mem.update(host.data())); }
  void copy_to(void *target) { iris::Task task;iris_check(task.flush_out(mem));iris_check(task.submit(iris_gpu,nullptr,true));memcpy(target,host.data(),host.size()); }
};
using LMemory=std::shared_ptr<LBuffer>;
