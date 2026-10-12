#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

template <typename T> __device__ float to_float_device(T value) {
  return static_cast<float>(value);
}
template <> __device__ float to_float_device<__half>(__half value) {
  return __half2float(value);
}
template <> __device__ float to_float_device<__hip_bfloat16>(__hip_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T> __device__ T from_float_device(float value) {
  return static_cast<T>(value);
}
template <> __device__ __half from_float_device<__half>(float value) {
  return __float2half(value);
}
template <> __device__ __hip_bfloat16 from_float_device<__hip_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T>
__device__ void add_bias_residual_layer_norm(T* out, const T* input, const T* bias,
                                             const T* gamma, const T* beta,
                                             float epsilon, int n) {
  __shared__ float partial[256];
  const int tid = threadIdx.x;
  const int base = blockIdx.x * n;
  float sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    const float value = to_float_device(out[base + i]) +
                        to_float_device(input[base + i]) +
                        to_float_device(bias[i]);
    out[base + i] = from_float_device<T>(value);
    sum += value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  const float mean = partial[0] / static_cast<float>(n);
  sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    const float value = to_float_device(out[base + i]);
    const float diff = value - mean;
    sum += diff * diff;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  const float inv_std = rsqrtf(partial[0] / static_cast<float>(n) + epsilon);
  for (int i = tid; i < n; i += blockDim.x) {
    const float value = (to_float_device(out[base + i]) - mean) * inv_std *
                        to_float_device(gamma[i]) + to_float_device(beta[i]);
    out[base + i] = from_float_device<T>(value);
  }
}

extern "C" __global__ void addBiasResidualLayerNorm_float(
    float* out, const float* input, const float* bias, const float* gamma,
    const float* beta, float epsilon, int n) {
  add_bias_residual_layer_norm(out, input, bias, gamma, beta, epsilon, n);
}

extern "C" __global__ void addBiasResidualLayerNorm_half(
    __half* out, const __half* input, const __half* bias, const __half* gamma,
    const __half* beta, float epsilon, int n) {
  add_bias_residual_layer_norm(out, input, bias, gamma, beta, epsilon, n);
}

extern "C" __global__ void addBiasResidualLayerNorm_bfloat16(
    __hip_bfloat16* out, const __hip_bfloat16* input, const __hip_bfloat16* bias,
    const __hip_bfloat16* gamma, const __hip_bfloat16* beta, float epsilon, int n) {
  add_bias_residual_layer_norm(out, input, bias, gamma, beta, epsilon, n);
}
