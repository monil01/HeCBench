#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

template <typename T>
__device__ __forceinline__ T relaxed_load(T* address)
{
#if defined(__HIPCC__)
  return __atomic_load_n(address, __ATOMIC_RELAXED);
#else
  return *address;
#endif
}

__device__ __forceinline__ long long atomic_min(long long* address, long long val)
{
  long long ret = relaxed_load(address);
  while (val < ret) {
    long long old = ret;
    if ((ret = atomicCAS(reinterpret_cast<unsigned long long*>(address),
                         static_cast<unsigned long long>(old),
                         static_cast<unsigned long long>(val))) == old)
      break;
  }
  return ret;
}

__device__ __forceinline__ unsigned long long atomic_min(unsigned long long* address,
                                                         unsigned long long val)
{
  unsigned long long ret = relaxed_load(address);
  while (val < ret) {
    unsigned long long old = ret;
    if ((ret = atomicCAS(address, old, val)) == old) break;
  }
  return ret;
}

__device__ __forceinline__ double atomic_min(double* address, double val)
{
  unsigned long long ret = relaxed_load(reinterpret_cast<unsigned long long*>(address));
  while (val < __longlong_as_double(ret)) {
    unsigned long long old = ret;
    if ((ret = atomicCAS(reinterpret_cast<unsigned long long*>(address), old,
                         __double_as_longlong(val))) == old)
      break;
  }
  return __longlong_as_double(ret);
}

__device__ __forceinline__ long long atomic_max(long long* address, long long val)
{
  long long ret = relaxed_load(address);
  while (val > ret) {
    long long old = ret;
    if ((ret = static_cast<long long>(
             atomicCAS(reinterpret_cast<unsigned long long*>(address),
                       static_cast<unsigned long long>(old),
                       static_cast<unsigned long long>(val)))) == old)
      break;
  }
  return ret;
}

__device__ __forceinline__ unsigned long long atomic_max(unsigned long long* address,
                                                         unsigned long long val)
{
  unsigned long long ret = relaxed_load(address);
  while (val > ret) {
    unsigned long long old = ret;
    if ((ret = atomicCAS(address, old, val)) == old) break;
  }
  return ret;
}

__device__ __forceinline__ double atomic_max(double* address, double val)
{
  unsigned long long ret = relaxed_load(reinterpret_cast<unsigned long long*>(address));
  while (val > __longlong_as_double(ret)) {
    unsigned long long old = ret;
    if ((ret = atomicCAS(reinterpret_cast<unsigned long long*>(address), old,
                         __double_as_longlong(val))) == old)
      break;
  }
  return __longlong_as_double(ret);
}

__device__ __forceinline__ long long atomic_add(long long* address, long long val)
{
  long long old;
  long long ret = relaxed_load(address);
  do {
    old = ret;
  } while ((ret = static_cast<long long>(
                atomicCAS(reinterpret_cast<unsigned long long*>(address),
                          static_cast<unsigned long long>(old),
                          static_cast<unsigned long long>(old + val)))) != old);
  return ret;
}

__device__ __forceinline__ unsigned long long atomic_add(unsigned long long* address,
                                                         unsigned long long val)
{
  unsigned long long old;
  unsigned long long ret = relaxed_load(address);
  do {
    old = ret;
  } while ((ret = atomicCAS(address, old, old + val)) != old);
  return ret;
}

__device__ __forceinline__ double atomic_add(double* address, double val)
{
  unsigned long long* ptr = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old;
  unsigned long long ret = relaxed_load(ptr);
  do {
    old = ret;
  } while ((ret = atomicCAS(ptr, old, __double_as_longlong(__longlong_as_double(old) + val))) !=
           old);
  return __longlong_as_double(ret);
}
