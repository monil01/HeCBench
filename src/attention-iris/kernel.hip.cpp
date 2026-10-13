#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256
#if defined(__GFX8__) || defined(__GFX9__)
#define IRIS_WARP_SIZE 64
#else
#define IRIS_WARP_SIZE 32
#endif

extern "C" __global__ void attention_kernel1(const float* key, const float* query,
                                              float* dot, float* exp_sum, int n, int d)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sum = 0.0f;
  for (int j = 0; j < d; ++j) sum += key[i * d + j] * query[j];
  dot[i] = sum;
  atomicAdd(exp_sum, __expf(sum));
}

extern "C" __global__ void attention_kernel2(const float* exp_sum, const float* dot,
                                              float* score, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) score[i] = __expf(dot[i]) / exp_sum[0];
}

extern "C" __global__ void attention_kernel3(const float* score, const float* value,
                                              float* output, int n, int d)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d) return;
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) sum += score[i] * value[i * d + j];
  output[j] = sum;
}

__device__ float block_sum(float value)
{
  __shared__ float scratch[BLOCK_SIZE];
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
    if (tid < offset) scratch[tid] += scratch[tid + offset];
    __syncthreads();
  }
  return scratch[0];
}

extern "C" __global__ void attention_kernel1_blockReduce(const float* key, const float* query,
                                                          float* dot, float* exp_sum, int n, int d)
{
  const int i = blockIdx.x;
  if (i >= n) return;
  float sum = 0.0f;
  for (int j = threadIdx.x; j < d; j += blockDim.x) sum += key[i * d + j] * query[j];
  sum = block_sum(sum);
  if (threadIdx.x == 0) {
    dot[i] = sum;
    atomicAdd(exp_sum, __expf(sum));
  }
}

__device__ float warp_sum(float value)
{
  for (int offset = IRIS_WARP_SIZE / 2; offset > 0; offset >>= 1)
    value += __shfl_xor(value, offset, IRIS_WARP_SIZE);
  return value;
}

extern "C" __global__ void attention_kernel1_warpReduce(const float* key, const float* query,
                                                         float* dot, float* exp_sum, int n, int d)
{
  const int lane = threadIdx.x % IRIS_WARP_SIZE;
  const int warp = threadIdx.x / IRIS_WARP_SIZE;
  const int i = blockIdx.x * (BLOCK_SIZE / IRIS_WARP_SIZE) + warp;
  if (i >= n) return;
  float sum = 0.0f;
  for (int j = lane; j < d; j += IRIS_WARP_SIZE) sum += key[i * d + j] * query[j];
  sum = warp_sum(sum);
  if (lane == 0) {
    dot[i] = sum;
    atomicAdd(exp_sum, __expf(sum));
  }
}

extern "C" __global__ void attention_kernel2_blockReduce(const float* exp_sum, const float* dot,
                                                          const float* value, float* output, int n, int d)
{
  const int j = blockIdx.x;
  if (j >= d) return;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    sum += (__expf(dot[i]) / exp_sum[0]) * value[i * d + j];
  sum = block_sum(sum);
  if (threadIdx.x == 0) output[j] = sum;
}

extern "C" __global__ void attention_kernel2_warpReduce(const float* exp_sum, const float* dot,
                                                         const float* value, float* output, int n, int d)
{
  const int lane = threadIdx.x % IRIS_WARP_SIZE;
  const int warp = threadIdx.x / IRIS_WARP_SIZE;
  const int j = blockIdx.x * (BLOCK_SIZE / IRIS_WARP_SIZE) + warp;
  if (j >= d) return;
  float sum = 0.0f;
  for (int i = lane; i < n; i += IRIS_WARP_SIZE)
    sum += (__expf(dot[i]) / exp_sum[0]) * value[i * d + j];
  sum = warp_sum(sum);
  if (lane == 0) output[j] = sum;
}

// Backend-portable fallback retaining all four benchmark selectors.
template <int Implementation>
__device__ void attention_fused(const float* key, const float* value,
                                const float* query, float* output, int n, int d)
{
  (void)Implementation;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d) return;
  float denominator = 0.0f;
  for (int i = 0; i < n; ++i) {
    float dot = 0.0f;
    for (int k = 0; k < d; ++k) dot += key[i * d + k] * query[k];
    denominator += __expf(dot);
  }
  float result = 0.0f;
  for (int i = 0; i < n; ++i) {
    float dot = 0.0f;
    for (int k = 0; k < d; ++k) dot += key[i * d + k] * query[k];
    result += (__expf(dot) / denominator) * value[i * d + j];
  }
  output[j] = result;
}

extern "C" __global__ void attention_fused_impl0(const float* key, const float* value,
                                                  const float* query, float* output, int n, int d)
{ attention_fused<0>(key, value, query, output, n, d); }
extern "C" __global__ void attention_fused_impl1(const float* key, const float* value,
                                                  const float* query, float* output, int n, int d)
{ attention_fused<1>(key, value, query, output, n, d); }
extern "C" __global__ void attention_fused_impl2(const float* key, const float* value,
                                                  const float* query, float* output, int n, int d)
{ attention_fused<2>(key, value, query, output, n, d); }
extern "C" __global__ void attention_fused_impl3(const float* key, const float* value,
                                                  const float* query, float* output, int n, int d)
{ attention_fused<3>(key, value, query, output, n, d); }
