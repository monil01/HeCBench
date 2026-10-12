#include <hip/hip_runtime.h>

#include <cstdint>

extern "C" __global__ void rmsnorm_fwd_kernel(const float* __restrict__ input,
                                              const float* __restrict__ gamma,
                                              float* __restrict__ output,
                                              int64_t inner_len,
                                              float epsilon)
{
  __shared__ float fallback_smem[1024];
  const int row = blockIdx.x;
  const float* input_row = input + static_cast<int64_t>(row) * inner_len;
  float* output_row = output + static_cast<int64_t>(row) * inner_len;

  float local_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < inner_len; col += blockDim.x) {
    const float val = input_row[col];
    local_sum += val * val;
  }
  fallback_smem[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      fallback_smem[threadIdx.x] += fallback_smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float norm_factor = rsqrtf(fallback_smem[0] / static_cast<float>(inner_len) + epsilon);
  for (int64_t col = threadIdx.x; col < inner_len; col += blockDim.x) {
    output_row[col] = input_row[col] * norm_factor * gamma[col];
  }
}
