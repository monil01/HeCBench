#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

template <typename T>
__device__ void fma_basic_impl(const T* a, const T* b, T* c, const int* in_indices,
                               const int* out_indices, int num_ops, int C, int N_A, int N_C)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_ops * C) return;
  int op_idx = idx / C;
  int ch_idx = idx % C;
  int in_idx = in_indices[op_idx];
  int out_idx = out_indices[op_idx];
  if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) return;
  c[out_idx * C + ch_idx] += a[in_idx * C + ch_idx] * b[ch_idx];
}

template <typename T>
__device__ void fma_rowwise_impl(const T* a, const T* b, T* c, const int* in_indices,
                                 const int* out_indices, int num_ops, int C, int N_A, int N_C)
{
  int op_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (op_idx >= num_ops) return;
  int in_idx = in_indices[op_idx];
  int out_idx = out_indices[op_idx];
  if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) return;
  for (int ch_idx = 0; ch_idx < C; ++ch_idx) {
    c[out_idx * C + ch_idx] += a[in_idx * C + ch_idx] * b[ch_idx];
  }
}

#define DEFINE_FMA_KERNELS(TYPE, SUFFIX) \
extern "C" __global__ void fma_basic_##SUFFIX(const TYPE* a, const TYPE* b, TYPE* c, \
                                              const int* in_indices, const int* out_indices, \
                                              int num_ops, int C, int N_A, int N_C) \
{ \
  fma_basic_impl<TYPE>(a, b, c, in_indices, out_indices, num_ops, C, N_A, N_C); \
} \
extern "C" __global__ void fma_rowwise_##SUFFIX(const TYPE* a, const TYPE* b, TYPE* c, \
                                                const int* in_indices, const int* out_indices, \
                                                int num_ops, int C, int N_A, int N_C) \
{ \
  fma_rowwise_impl<TYPE>(a, b, c, in_indices, out_indices, num_ops, C, N_A, N_C); \
}

DEFINE_FMA_KERNELS(__half, half)
DEFINE_FMA_KERNELS(__hip_bfloat16, bfloat16)
DEFINE_FMA_KERNELS(float, float)
DEFINE_FMA_KERNELS(double, double)
