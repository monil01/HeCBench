#include <hip/hip_runtime.h>

extern "C" __global__ void dense_esuhm(const float* __restrict__ input,
                                        const float* __restrict__ dense,
                                        float* output,
                                        int embedding_dim,
                                        const int* __restrict__ offset)
{
  const int batch_idx = blockIdx.x;
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = offset[batch_idx + 1] - offset[batch_idx];
  for (int idx = tid; idx < embedding_dim; idx += grain_size) {
    const float dense_elem = dense[batch_idx * embedding_dim + idx];
    for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[offset[batch_idx] + nested_idx] = input[offset[batch_idx] + nested_idx] + dense_elem;
    }
  }
}

extern "C" __global__ void dense_esuhm2(const float* __restrict__ input,
                                         const float* __restrict__ dense,
                                         float* output,
                                         int embedding_dim,
                                         const int* __restrict__ offset)
{
  const int batch_idx = blockIdx.x;
  const int start = offset[batch_idx];
  const int range = offset[batch_idx + 1] - start;
  for (int idx = threadIdx.x; idx < embedding_dim; idx += blockDim.x) {
    const float dense_elem = dense[batch_idx * embedding_dim + idx];
    for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[start + nested_idx] = input[start + nested_idx] + dense_elem;
    }
  }
}

extern "C" __global__ void dense_esuhm3(const float* __restrict__ input,
                                         const float* __restrict__ dense,
                                         float* output,
                                         int embedding_dim,
                                         const int* __restrict__ offset)
{
  const int batch_idx = blockIdx.x;
  const int start = offset[batch_idx];
  const int range = offset[batch_idx + 1] - start;

  for (int s = 0; s < range; s += blockDim.x) {
    int idx = s + threadIdx.x;
    if (idx < range) {
      float input_elem = input[start + idx];
      float dense_elem = dense[batch_idx * embedding_dim + idx % embedding_dim];
      output[start + idx] = input_elem + dense_elem;
    }
  }
}
