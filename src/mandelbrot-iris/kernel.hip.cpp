#include <hip/hip_runtime.h>

struct ComplexF {
  float real;
  float imag;
};

__device__ float scale_row(int i, int rows)
{
  return -1.5f + (i * (2.0f / rows));
}

__device__ float scale_col(int i, int cols)
{
  return -1.0f + (i * (2.0f / cols));
}

__device__ int mandel_point(const ComplexF& c, int max_iterations)
{
  int count = 0;
  ComplexF z = {0.0f, 0.0f};
  for (int i = 0; i < max_iterations; ++i) {
    const float r = z.real;
    const float im = z.imag;
    if (((r * r) + (im * im)) >= 4.0f) {
      break;
    }
    z.real = r * r - im * im + c.real;
    z.imag = 2.0f * r * im + c.imag;
    count++;
  }
  return count;
}

extern "C" __global__ void mandel(int* out, int rows, int cols, int max_iterations)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < rows && j < cols) {
    ComplexF c = {scale_row(i, rows), scale_col(j, cols)};
    out[i * cols + j] = mandel_point(c, max_iterations);
  }
}
