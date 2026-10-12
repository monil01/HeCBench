#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void ReluGrad_impl1(
    const half* __restrict__ gradient,
    const half* __restrict__ feature,
    half* __restrict__ backprop,
    const int count)
{
  int half2_count = count >> 1;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_device_threads = gridDim.x * blockDim.x;

  while (index < half2_count) {
    half2 gradient_h2 = reinterpret_cast<const half2*>(gradient)[index];
    half2 feature_h2 = reinterpret_cast<const half2*>(feature)[index];
    half2* p_backprop_h2 = reinterpret_cast<half2*>(backprop) + index;
    const half2 kZero_h2 = __float2half2_rn(0.f);
    half2 mask_h2 = __hgt2(feature_h2, kZero_h2);
    half2 backprop_h2 = __hmul2(mask_h2, gradient_h2);
    *p_backprop_h2 = backprop_h2;
    index += total_device_threads;
  }

  if ((count & 0x1) == 1 && index == half2_count) {
    half grad_h = gradient[count - 1];
    half feature_h = feature[count - 1];
    float grad_f = static_cast<float>(grad_h);
    float feature_f = static_cast<float>(feature_h);
    float backprop_f = (feature_f > 0) ? grad_f : 0;
    half backprop_h(backprop_f);
    backprop[count - 1] = backprop_h;
  }
}

static constexpr int VectorSize = 8;

extern "C" __global__ void ReluGrad_impl2(
    const half* __restrict__ gradient,
    const half* __restrict__ feature,
    half* __restrict__ backprop,
    const int count)
{
  int half8_count = count / VectorSize;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < half8_count) {
    float4 gradient_h8 = reinterpret_cast<const float4*>(gradient)[index];
    float4 feature_h8 = reinterpret_cast<const float4*>(feature)[index];
    float4* p_backprop_h8 = reinterpret_cast<float4*>(backprop) + index;
    half2 *gradient_h2 = reinterpret_cast<half2*>(&gradient_h8);
    half2 *feature_h2 = reinterpret_cast<half2*>(&feature_h8);
    float4 backprop_h8;
    half2* p_backprop_h2 = reinterpret_cast<half2*>(&backprop_h8);
    const half2 kZero_h2 = __float2half2_rn(0.f);
    for (int i = 0; i < VectorSize / 2; i++) {
      half2 mask_h2 = __hgt2(feature_h2[i], kZero_h2);
      half2 backprop_h2 = __hmul2(mask_h2, gradient_h2[i]);
      p_backprop_h2[i] = backprop_h2;
    }
    *p_backprop_h8 = backprop_h8;
  }

  int remaining_count = (count % VectorSize);
  if (index < remaining_count) {
    half grad_h = gradient[half8_count * VectorSize + index];
    half feature_h = feature[half8_count * VectorSize + index];
    float grad_f = static_cast<float>(grad_h);
    float feature_f = static_cast<float>(feature_h);
    float backprop_f = (feature_f > 0) ? grad_f : 0;
    half backprop_h(backprop_f);
    backprop[half8_count * VectorSize + index] = backprop_h;
  }
}

extern "C" __global__ void Relu_impl1(int count, const int* input, int* output)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    uint4 v;
    char4 b;
    b.x = input[index] & 0xFF;
    b.y = (input[index] >> 8) & 0xFF;
    b.z = (input[index] >> 16) & 0xFF;
    b.w = (input[index] >> 24) & 0xFF;
    v.x = max(b.x, 0);
    v.y = max(b.y, 0);
    v.z = max(b.z, 0);
    v.w = max(b.w, 0);
    output[index] = v.w << 24 | v.z << 16 | v.y << 8 | v.x;
  }
}
