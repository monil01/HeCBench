#include <hip/hip_runtime.h>

template <typename T>
__device__ void sortkv_fill(int* keys, T* vals, int n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(n)) return;
  keys[i] = static_cast<int>(i);
  vals[i] = static_cast<T>(i % 256);
}

extern "C" __global__ void sortkv_u8(int* keys, unsigned char* vals, int n) {
  sortkv_fill(keys, vals, n);
}

extern "C" __global__ void sortkv_i16(int* keys, short* vals, int n) {
  sortkv_fill(keys, vals, n);
}

extern "C" __global__ void sortkv_i32(int* keys, int* vals, int n) {
  sortkv_fill(keys, vals, n);
}

extern "C" __global__ void sortkv_i64(int* keys, long* vals, int n) {
  sortkv_fill(keys, vals, n);
}
