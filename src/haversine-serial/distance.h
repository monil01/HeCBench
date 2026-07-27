#ifndef DIST_H
#define DIST_H

// Local serial variant of distance.h. The one in ../haversine-cuda insists
// on <sycl/sycl.hpp> when neither CUDA/HIP/_OPENMP is defined, which g++
// can't satisfy. This file provides the same interface without the SYCL
// dependency.

#include <stdio.h>
#include <math.h>
#include <chrono>

typedef struct __attribute__((__aligned__(16)))
{
  double x, y, z, w;
} double4;

#define DEGREE_TO_RADIAN  M_PI / 180.0
#define RADIAN_TO_DEGREE  180.0 / M_PI
#define EARTH_RADIUS_KM   6371.0

#endif
