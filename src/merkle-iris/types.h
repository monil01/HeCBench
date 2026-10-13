#pragma once
#include <cstdint>
#include <cstddef>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#define __device__
typedef unsigned int uint;
struct alignas(32) ulong4 {unsigned long x,y,z,w;};
inline ulong4 make_ulong4(unsigned long x,unsigned long y,unsigned long z,unsigned long w) {return {x,y,z,w};}
#endif
typedef unsigned long ulong;
static_assert(sizeof(ulong)==8 && sizeof(ulong4)==32,"original 64-bit field/vector ABI");
