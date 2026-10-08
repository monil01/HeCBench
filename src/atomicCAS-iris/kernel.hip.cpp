#include "kernels_common.h"

extern "C" __global__ void atomic_min_u64(unsigned long long* res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_min(res, static_cast<unsigned long long>(i));
}

extern "C" __global__ void atomic_min_s64(long long* res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_min(res, static_cast<long long>(i));
}

extern "C" __global__ void atomic_min_f64(double* res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_min(res, static_cast<double>(i));
}

extern "C" __global__ void atomic_max_u64(unsigned long long* res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_max(res, static_cast<unsigned long long>(i));
}

extern "C" __global__ void atomic_max_s64(long long* res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_max(res, static_cast<long long>(i));
}

extern "C" __global__ void atomic_max_f64(double* res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_max(res, static_cast<double>(i));
}

extern "C" __global__ void atomic_add_u64(unsigned long long* res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_add(res, static_cast<unsigned long long>(i));
}

extern "C" __global__ void atomic_add_s64(long long* res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_add(res, static_cast<long long>(i));
}

extern "C" __global__ void atomic_add_f64(double* res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_add(res, static_cast<double>(i));
}
