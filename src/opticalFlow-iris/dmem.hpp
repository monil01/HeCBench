#pragma once
#include <iris/iris.hpp>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
inline void iris_check(int rc) {
  if(rc != IRIS_SUCCESS) { fprintf(stderr,"IRIS error %d\n",rc); exit(1); }
}
struct FlowBuffer {
  std::vector<float> host;
  iris::DMem mem;
  explicit FlowBuffer(size_t count, const float *input=nullptr)
      : host(count), mem(host.data(),count*sizeof(float)) {
    if(input) { std::memcpy(host.data(),input,count*sizeof(float)); iris_check(mem.update(host.data())); }
  }
  void zero() { std::fill(host.begin(),host.end(),0.f); iris_check(mem.update(host.data())); }
};
