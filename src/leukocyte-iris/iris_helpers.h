#pragma once
#include <iris/iris.hpp>
#include <memory>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
inline void iris_check(int rc) {if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS task/memory failure");}
using MemoryOwners=std::vector<std::unique_ptr<iris::DMem>>;
inline iris::DMem& bind_memory(MemoryOwners& owners,void* host,size_t bytes){
 owners.emplace_back(std::make_unique<iris::DMem>(host,bytes));return *owners.back();
}
inline void require_hip_gpu(){
 int n=0;iris_check(iris_device_count(&n));
 for(int d=0;d<n;++d){int backend=0,type=0;char name[256]={};
  iris_check(iris_device_info(d,iris_backend,&backend,nullptr));
  iris_check(iris_device_info(d,iris_type,&type,nullptr));
  iris_check(iris_device_info(d,iris_name,name,nullptr));
  printf("IRIS backend=%d type=%d name=%s\n",backend,type,name);
  if(backend==iris_hip&&(type&iris_gpu)){iris_check(iris_device_set_default(d));return;}
 }
 throw std::runtime_error("actual IRIS HIP GPU required");
}
