#include <cuda.h>

#define __shfl(v, d) __shfl_sync(0xffffffff, v, d)
#define __shfl_xor(v, d) __shfl_xor_sync(0xffffffff, v, d)

extern "C" __global__ void bcast_shfl_sg8(const int arg, int* out) {
  int value = ((threadIdx.x & 0x7) == 0) ? arg : 0;
  int out_v = __shfl(value, 0);
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = out_v;
}

extern "C" __global__ void bcast_shfl_xor_sg8(int* out) {
  int value = (threadIdx.x & 0x7);
  for (int mask = 1; mask < 0x7; mask *= 2) value += __shfl_xor(value, mask);
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = value;
}

extern "C" __global__ void bcast_shfl_sg16(const int arg, int* out) {
  int value = ((threadIdx.x & 0xf) == 0) ? arg : 0;
  int out_v = __shfl(value, 0);
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = out_v;
}

extern "C" __global__ void bcast_shfl_xor_sg16(int* out) {
  int value = (threadIdx.x & 0xf);
  for (int mask = 1; mask < 0xf; mask *= 2) value += __shfl_xor(value, mask);
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = value;
}

extern "C" __global__ void bcast_shfl_sg32(const int arg, int* out) {
  int value = ((threadIdx.x & 0x1f) == 0) ? arg : 0;
  int out_v = __shfl(value, 0);
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = out_v;
}

extern "C" __global__ void bcast_shfl_xor_sg32(int* out) {
  int value = (threadIdx.x & 0x1f);
  for (int mask = 1; mask < 0x1f; mask *= 2) value += __shfl_xor(value, mask);
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = value;
}

extern "C" __global__ void transpose_shfl(float* out, const float* in) {
  unsigned b_start = blockDim.x * blockIdx.x;
  unsigned b_offs = b_start + threadIdx.x;
  unsigned s_offs = blockDim.x - threadIdx.x - 1;
  float val = in[b_offs];
  out[b_offs] = __shfl(val, s_offs);
}
