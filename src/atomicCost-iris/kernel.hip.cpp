#include <hip/hip_runtime.h>

template <typename T>
__device__ void woAtomicOnGlobalMem(T* result, int size)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid * size; i < (tid + 1) * size; i++) {
    result[tid] += i % 2;
  }
}

template <typename T>
__device__ void wiAtomicOnGlobalMem(T* result, int size)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid * size; i < (tid + 1) * size; i++) {
    atomicAdd(&result[tid], static_cast<T>(i % 2));
  }
}

extern "C" __global__ void woAtomicOnGlobalMem_f64(double* result, int size)
{
  woAtomicOnGlobalMem(result, size);
}

extern "C" __global__ void wiAtomicOnGlobalMem_f64(double* result, int size)
{
  wiAtomicOnGlobalMem(result, size);
}

extern "C" __global__ void woAtomicOnGlobalMem_i32(int* result, int size)
{
  woAtomicOnGlobalMem(result, size);
}

extern "C" __global__ void wiAtomicOnGlobalMem_i32(int* result, int size)
{
  wiAtomicOnGlobalMem(result, size);
}

extern "C" __global__ void woAtomicOnGlobalMem_f32(float* result, int size)
{
  woAtomicOnGlobalMem(result, size);
}

extern "C" __global__ void wiAtomicOnGlobalMem_f32(float* result, int size)
{
  wiAtomicOnGlobalMem(result, size);
}
