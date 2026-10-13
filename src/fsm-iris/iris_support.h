#pragma once
#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
inline void iris_check(int status, const char* operation) {
  if (status != IRIS_SUCCESS) throw std::runtime_error(operation);
}
inline int iris_hip_device() {
  int count=0, selected=-1;
  iris_check(iris_device_count(&count), "device count");
  for (int i=0; i<count; ++i) {
    int backend=-1, type=0; char name[256]={};
    iris_check(iris_device_info(i, iris_backend, &backend, nullptr), "backend");
    iris_check(iris_device_info(i, iris_type, &type, nullptr), "device type");
    iris_check(iris_device_info(i, iris_name, name, nullptr), "device name");
    printf("IRIS device %d: %s backend=%d type=%d\n", i,name,backend,type);
    if (backend==iris_hip && (type & iris_gpu) && selected<0) selected=i;
  }
  if (selected<0) throw std::runtime_error("No HIP GPU available");
  printf("Selected HIP GPU device %d\n", selected);
  return selected;
}
