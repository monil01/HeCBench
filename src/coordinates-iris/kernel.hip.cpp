#include <hip/hip_runtime.h>

#include "utils.hpp"

extern "C" __global__ void coordinates_transform_double(
    const lonlat_2d<double>* input,
    cartesian_2d<double>* output,
    int num_coords,
    double origin_x,
    double origin_y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_coords) {
    lonlat_2d<double> origin;
    origin.x = origin_x;
    origin.y = origin_y;
    output[i] = to_cartesian_functor<double>(origin)(input[i]);
  }
}

extern "C" __global__ void coordinates_transform_float(
    const lonlat_2d<float>* input,
    cartesian_2d<float>* output,
    int num_coords,
    float origin_x,
    float origin_y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_coords) {
    lonlat_2d<float> origin;
    origin.x = origin_x;
    origin.y = origin_y;
    output[i] = to_cartesian_functor<float>(origin)(input[i]);
  }
}
