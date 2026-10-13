#pragma once
#include <iris/iris.hpp>
#include <stdexcept>
#include <iostream>
inline void check_iris(int status, const char* stage="IRIS operation") {
  if (status != IRIS_SUCCESS) throw std::runtime_error(std::string(stage)+" failed status="+std::to_string(status));
}
inline int init_iris(iris::Platform& platform, int& argc, char**& argv) {
  check_iris(platform.init(&argc, &argv, true), "platform.init");
  int count = 0, selected = -1;
  check_iris(iris_device_count(&count));
  for (int i = 0; i < count; ++i) {
    int type = 0; char name[256] = {};
    check_iris(iris_device_info(i, iris_type, &type, nullptr));
    check_iris(iris_device_info(i, iris_name, name, nullptr));
    std::cout << "IRIS device " << i << ": " << name << " type=" << type << '\n';
    if ((type & iris_amd) && selected < 0) selected = i;
  }
  if (selected < 0) throw std::runtime_error("No IRIS AMD GPU");
  std::cout << "Selected IRIS AMD device " << selected << '\n';
  return selected;
}
inline void launch_iris(int device, const char* name, size_t global, size_t local,
                        int count, void** params, int* info, iris::DMem* output = nullptr) {
  iris::Task task;
  check_iris(task.kernel(name, 1, nullptr, &global, &local, count, params, info), name);
  if (output) check_iris(task.flush_out(*output), "task.flush_out");
  check_iris(task.submit(device, nullptr, true), "task.submit");
}
inline void flush_iris(int device, iris::DMem& output) {
  iris::Task task;
  check_iris(task.flush_out(output), "task.flush_out");
  check_iris(task.submit(device, nullptr, true));
}
