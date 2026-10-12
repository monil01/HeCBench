#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void tensor_packed_accessor_kernel(const int64_t nrow,
                                                          const int64_t ncol,
                                                          float* __restrict__ r,
                                                          const float* __restrict__ m,
                                                          const float* __restrict__ v)
{
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nrow) {
    float val = 0.0f;
    for (int64_t j = 0; j < ncol; j++) {
      val += m[i * ncol + j] * v[j];
    }
    r[i] = val;
  }
}

extern "C" __global__ void raw_accessor_kernel(const int64_t nrow,
                                                const int64_t ncol,
                                                float* __restrict__ r,
                                                const float* __restrict__ m,
                                                const float* __restrict__ v)
{
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nrow) {
    float val = 0.0f;
    for (int64_t j = 0; j < ncol; j++) {
      val += m[i * ncol + j] * v[j];
    }
    r[i] = val;
  }
}
