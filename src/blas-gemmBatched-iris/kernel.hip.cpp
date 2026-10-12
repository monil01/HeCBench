#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

template <typename T>
__device__ T gemm_accum(const T* matrices, const T* vectors,
                       int upper, int size, int batch, int row)
{
  T sum = static_cast<T>(0);
  const int matrix_base = batch * upper * upper;
  const int vector_base = batch * upper;
  for (int k = 0; k < size; ++k) {
    sum += matrices[matrix_base + row + k * upper] * vectors[vector_base + k];
  }
  return sum;
}

extern "C" __global__ void gemm_batched_float(const float* matrices,
                                               const float* vectors,
                                               float* result,
                                               int upper,
                                               int size,
                                               int num)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num * size;
  if (idx >= total) return;

  const int batch = idx / size;
  const int row = idx - batch * size;
  result[batch * upper + row] =
      gemm_accum<float>(matrices, vectors, upper, size, batch, row);
}

extern "C" __global__ void gemm_batched_double(const double* matrices,
                                                const double* vectors,
                                                double* result,
                                                int upper,
                                                int size,
                                                int num)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num * size;
  if (idx >= total) return;

  const int batch = idx / size;
  const int row = idx - batch * size;
  result[batch * upper + row] =
      gemm_accum<double>(matrices, vectors, upper, size, batch, row);
}

extern "C" __global__ void gemm_batched_half(const __half* matrices,
                                              const __half* vectors,
                                              __half* result,
                                              int upper,
                                              int size,
                                              int num)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num * size;
  if (idx >= total) return;

  const int batch = idx / size;
  const int row = idx - batch * size;
  float sum = 0.0f;
  const int matrix_base = batch * upper * upper;
  const int vector_base = batch * upper;
  for (int k = 0; k < size; ++k) {
    sum += __half2float(matrices[matrix_base + row + k * upper]) *
           __half2float(vectors[vector_base + k]);
  }
  result[batch * upper + row] = __float2half(sum);
}
