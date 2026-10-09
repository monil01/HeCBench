#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256

template <typename T>
__device__ double block_dot(const T* a, const T* b, int n)
{
  __shared__ double cache[BLOCK_SIZE];
  const int tid = static_cast<int>(threadIdx.x);
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  double value = 0.0;
  if (idx < n) value = static_cast<double>(a[idx]) * static_cast<double>(b[idx]);
  cache[tid] = value;
  __syncthreads();

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) cache[tid] += cache[tid + stride];
    __syncthreads();
  }
  return cache[0];
}

extern "C" __global__ void dot_partial_f64(double* partial,
                                            const double* a,
                                            const double* b,
                                            int n)
{
  const double sum = block_dot(a, b, n);
  if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

extern "C" __global__ void dot_partial_f32(double* partial,
                                            const float* a,
                                            const float* b,
                                            int n)
{
  const double sum = block_dot(a, b, n);
  if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

extern "C" __global__ void dot_partial_f16(double* partial,
                                            const __half* a,
                                            const __half* b,
                                            int n)
{
  __shared__ double cache[BLOCK_SIZE];
  const int tid = static_cast<int>(threadIdx.x);
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  double value = 0.0;
  if (idx < n) {
    value = static_cast<double>(__half2float(a[idx])) *
            static_cast<double>(__half2float(b[idx]));
  }
  cache[tid] = value;
  __syncthreads();
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) cache[tid] += cache[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = cache[0];
}

extern "C" __global__ void dot_partial_bf16(double* partial,
                                             const __hip_bfloat16* a,
                                             const __hip_bfloat16* b,
                                             int n)
{
  __shared__ double cache[BLOCK_SIZE];
  const int tid = static_cast<int>(threadIdx.x);
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  double value = 0.0;
  if (idx < n) {
    value = static_cast<double>(__bfloat162float(a[idx])) *
            static_cast<double>(__bfloat162float(b[idx]));
  }
  cache[tid] = value;
  __syncthreads();
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) cache[tid] += cache[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = cache[0];
}

extern "C" __global__ void dot_final_sum(double* dst,
                                          const double* partial,
                                          int num_blocks)
{
  __shared__ double cache[BLOCK_SIZE];
  const int tid = static_cast<int>(threadIdx.x);
  double sum = 0.0;
  for (int i = tid; i < num_blocks; i += BLOCK_SIZE) {
    sum += partial[i];
  }
  cache[tid] = sum;
  __syncthreads();
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) cache[tid] += cache[tid + stride];
    __syncthreads();
  }
  if (tid == 0) *dst = cache[0];
}
