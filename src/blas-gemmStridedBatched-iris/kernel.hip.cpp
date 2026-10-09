#include <hip/hip_runtime.h>

extern "C" __global__ void gemm_batched_float(const float* matrices,
                                               const float* vectors,
                                               float* result,
                                               int num,
                                               int upper,
                                               int m,
                                               int k,
                                               float alpha,
                                               float beta)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num * m;
  if (idx >= total) {
    return;
  }
  const int batch = idx / m;
  const int row = idx - batch * m;
  float sum = 0.0f;
  for (int col = 0; col < k; col++) {
    sum += matrices[batch * upper * upper + row + col * upper] *
           vectors[batch * upper + col];
  }
  result[batch * upper + row] = alpha * sum + beta * result[batch * upper + row];
}

extern "C" __global__ void gemm_batched_double(const double* matrices,
                                                const double* vectors,
                                                double* result,
                                                int num,
                                                int upper,
                                                int m,
                                                int k,
                                                double alpha,
                                                double beta)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num * m;
  if (idx >= total) {
    return;
  }
  const int batch = idx / m;
  const int row = idx - batch * m;
  double sum = 0.0;
  for (int col = 0; col < k; col++) {
    sum += matrices[batch * upper * upper + row + col * upper] *
           vectors[batch * upper + col];
  }
  result[batch * upper + row] = alpha * sum + beta * result[batch * upper + row];
}
