#include <hip/hip_runtime.h>

#include <cstdint>
#include <cmath>

extern "C" __global__ void silu_forward_kernel(const float* x, float* out, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  float x_val = x[i];
  out[i] = x_val / (1.0f + expf(-x_val));
}

extern "C" __global__ void silu_forward_kernel2(const float* x, float* out, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t vec_count = N / 4;
  if (i < vec_count) {
    float4 x4 = reinterpret_cast<const float4*>(x)[i];
    float4 t4;
    t4.x = x4.x / (1.0f + expf(-x4.x));
    t4.y = x4.y / (1.0f + expf(-x4.y));
    t4.z = x4.z / (1.0f + expf(-x4.z));
    t4.w = x4.w / (1.0f + expf(-x4.w));
    reinterpret_cast<float4*>(out)[i] = t4;
  }
  for (uint64_t j = 4 * vec_count + i; j < N; j += blockDim.x) {
    float x_val = x[j];
    out[j] = x_val / (1.0f + expf(-x_val));
  }
}

extern "C" __global__ void silu_backward_kernel(const float* dout, const float* x, float* dx, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  float out_val = dout[i];
  float x_val = x[i];
  float expx = expf(-x_val);
  float grad_silu = (1.0f + x_val * expx / (1.0f + expx)) / (1.0f + expx);
  dx[i] = out_val * grad_silu;
}

extern "C" __global__ void silu_backward_kernel2(const float* dout, const float* x, float* dx, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  float x_val = x[i];
  float sig = 1.0f / (1.0f + expf(-x_val));
  float grad_silu = sig * (1.0f + x_val * (1.0f - sig));
  dx[i] = dout[i] * grad_silu;
}

extern "C" __global__ void silu_backward_kernel3(const float* dout, const float* x, float* dx, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t vec_count = N / 4;
  if (i < vec_count) {
    float4 x4 = reinterpret_cast<const float4*>(x)[i];
    float4 do4 = reinterpret_cast<const float4*>(dout)[i];
    float4 sig4, grad_silu4;
    sig4.x = 1.0f / (1.0f + expf(-x4.x));
    sig4.y = 1.0f / (1.0f + expf(-x4.y));
    sig4.z = 1.0f / (1.0f + expf(-x4.z));
    sig4.w = 1.0f / (1.0f + expf(-x4.w));
    grad_silu4.x = sig4.x * (1.0f + x4.x * (1.0f - sig4.x));
    grad_silu4.y = sig4.y * (1.0f + x4.y * (1.0f - sig4.y));
    grad_silu4.z = sig4.z * (1.0f + x4.z * (1.0f - sig4.z));
    grad_silu4.w = sig4.w * (1.0f + x4.w * (1.0f - sig4.w));
    sig4.x = do4.x * grad_silu4.x;
    sig4.y = do4.y * grad_silu4.y;
    sig4.z = do4.z * grad_silu4.z;
    sig4.w = do4.w * grad_silu4.w;
    reinterpret_cast<float4*>(dx)[i] = sig4;
  }
  for (uint64_t j = 4 * vec_count + i; j < N; j += blockDim.x) {
    float x_val = x[j];
    float sig = 1.0f / (1.0f + expf(-x_val));
    float grad_silu = sig * (1.0f + x_val * (1.0f - sig));
    dx[j] = dout[j] * grad_silu;
  }
}
