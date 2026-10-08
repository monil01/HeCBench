#include <cuda_runtime.h>

extern "C" __global__ void rotate_matrix_parallel(float* matrix, const int n)
{
  int layer = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer < n / 2) {
    int first = layer;
    int last = n - 1 - layer;
    for (int i = first; i < last; ++i) {
      int offset = i - first;
      float top = matrix[first * n + i];
      matrix[first * n + i] = matrix[(last - offset) * n + first];
      matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];
      matrix[last * n + (last - offset)] = matrix[i * n + last];
      matrix[i * n + last] = top;
    }
  }
}
