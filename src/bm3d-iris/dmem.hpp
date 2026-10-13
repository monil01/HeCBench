#pragma once
#include <iris/iris.hpp>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
inline void iris_check(int rc) { if(rc!=IRIS_SUCCESS) { fprintf(stderr,"IRIS error %d\n",rc); exit(1); } }
struct BMBuffer {
  std::vector<std::uint8_t> host;
  iris::DMem mem;
  explicit BMBuffer(size_t bytes,void *external=nullptr)
    : host(external ? 0 : bytes),mem(external ? external : host.data(),bytes) {}
  void zero() { memset(host.data(),0,host.size());iris_check(mem.update(host.data())); }
};
