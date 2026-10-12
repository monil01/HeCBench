#include <hip/hip_runtime.h>

static __device__ __forceinline__ void scatter_add_direct(
    const int* indexes,
    const float* source,
    float* outputs,
    int batch_size,
    int output_size,
    int vector_dim)
{
  const int local = threadIdx.y * blockDim.x + threadIdx.x;
  const int threads_per_block = blockDim.x * blockDim.y;
  const int block_start = blockIdx.x * threads_per_block;
  const int stride = gridDim.x * threads_per_block;
  const int total = batch_size * vector_dim;

  for (int linear = block_start + local; linear < total; linear += stride) {
    const int batch_idx = linear / vector_dim;
    const int dim = linear - batch_idx * vector_dim;
    const int index = indexes[batch_idx];
    if (index >= 0 && index < output_size) {
      atomicAdd(&outputs[index * vector_dim + dim], source[linear]);
    }
  }
}

extern "C" __global__ void scatterAdd_kernel(
    const int* indexes,
    const float* source,
    float* outputs,
    const int batch_size,
    const int output_size,
    const int vector_dim)
{
  scatter_add_direct(indexes, source, outputs, batch_size, output_size, vector_dim);
}

extern "C" __global__ void scatterAdd2_kernel(
    const int* indexes,
    const float* source,
    float* outputs,
    const int batch_size,
    const int output_size,
    const int vector_dim)
{
  scatter_add_direct(indexes, source, outputs, batch_size, output_size, vector_dim);
}
