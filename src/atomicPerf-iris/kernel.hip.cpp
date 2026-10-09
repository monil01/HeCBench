#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256

template <typename T>
__device__ void block_range_global(T* data, int n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid; i < static_cast<unsigned int>(n);
       i += blockDim.x * gridDim.x) {
    atomicAdd(data + threadIdx.x, static_cast<T>(1));
  }
}

template <typename T>
__device__ void warp_range_global(T* data, int n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid; i < static_cast<unsigned int>(n);
       i += blockDim.x * gridDim.x) {
    atomicAdd(data + (i & 0x1F), static_cast<T>(1));
  }
}

template <typename T>
__device__ void single_range_global(T* data, int offset, int n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid; i < static_cast<unsigned int>(n);
       i += blockDim.x * gridDim.x) {
    atomicAdd(data + offset, static_cast<T>(1));
  }
}

template <typename T>
__device__ void block_range_shared(T* data, int n) {
  __shared__ T smem_data[BLOCK_SIZE];
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid; i < static_cast<unsigned int>(n);
       i += blockDim.x * gridDim.x) {
    atomicAdd(smem_data + threadIdx.x, static_cast<T>(1));
  }
  if (blockIdx.x == gridDim.x) data[threadIdx.x] = smem_data[threadIdx.x];
}

template <typename T>
__device__ void warp_range_shared(T* data, int n) {
  __shared__ T smem_data[32];
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid; i < static_cast<unsigned int>(n);
       i += blockDim.x * gridDim.x) {
    atomicAdd(smem_data + (i & 0x1F), static_cast<T>(1));
  }
  if (blockIdx.x == gridDim.x && threadIdx.x < 0x1F) {
    data[threadIdx.x] = smem_data[threadIdx.x];
  }
}

template <typename T>
__device__ void single_range_shared(T* data, int offset, int n) {
  __shared__ T smem_data[BLOCK_SIZE];
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid; i < static_cast<unsigned int>(n);
       i += blockDim.x * gridDim.x) {
    atomicAdd(smem_data + offset, static_cast<T>(1));
  }
  if (blockIdx.x == gridDim.x && threadIdx.x == 0) data[0] = smem_data[0];
}

#define MAKE_KERNELS(T, suffix)                                                  \
  extern "C" __global__ void BlockRangeAtomicOnGlobalMem_##suffix(              \
      T* data, int, int n) {                                                     \
    block_range_global<T>(data, n);                                              \
  }                                                                              \
  extern "C" __global__ void WarpRangeAtomicOnGlobalMem_##suffix(               \
      T* data, int, int n) {                                                     \
    warp_range_global<T>(data, n);                                               \
  }                                                                              \
  extern "C" __global__ void SingleRangeAtomicOnGlobalMem_##suffix(             \
      T* data, int offset, int n) {                                              \
    single_range_global<T>(data, offset, n);                                     \
  }                                                                              \
  extern "C" __global__ void BlockRangeAtomicOnSharedMem_##suffix(              \
      T* data, int, int n) {                                                     \
    block_range_shared<T>(data, n);                                              \
  }                                                                              \
  extern "C" __global__ void WarpRangeAtomicOnSharedMem_##suffix(               \
      T* data, int, int n) {                                                     \
    warp_range_shared<T>(data, n);                                               \
  }                                                                              \
  extern "C" __global__ void SingleRangeAtomicOnSharedMem_##suffix(             \
      T* data, int offset, int n) {                                              \
    single_range_shared<T>(data, offset, n);                                     \
  }

MAKE_KERNELS(double, double)
MAKE_KERNELS(int, int)
MAKE_KERNELS(float, float)
