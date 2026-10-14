#pragma once
#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
static int hip_device = -1;
static void iris_check(int rc, const char* operation) {
  if (rc != IRIS_SUCCESS) {
    std::fprintf(stderr, "IRIS %s failed: %d\n", operation, rc);
    std::exit(2);
  }
}
static void select_hip_device() {
  int count = 0;
  iris_check(iris_device_count(&count), "device count");
  for (int dev = 0; dev < count; ++dev) {
    int type = 0, backend = -1;
    char name[256] = {};
    iris_check(iris_device_info(dev, iris_type, &type, nullptr), "device type");
    iris_check(iris_device_info(dev, iris_backend, &backend, nullptr), "backend");
    iris_check(iris_device_info(dev, iris_name, name, nullptr), "device name");
    if ((type & iris_amd) && backend == 3 && hip_device < 0) {
      hip_device = dev;
      std::fprintf(stderr, "Selected actual HIP GPU %d: %s backend=%d\n", dev, name, backend);
    }
  }
  if (hip_device < 0) {
    std::fprintf(stderr, "No actual HIP AMD GPU available\n");
    std::exit(2);
  }
}
