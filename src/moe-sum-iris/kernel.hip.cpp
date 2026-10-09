#include <hip/hip_runtime.h>

template <int TOPK>
__device__ void moe_sum_body(float* __restrict__ out, const float* __restrict__ input, int d)
{
  const int64_t output_base = static_cast<int64_t>(blockIdx.x) * d;
  const int64_t input_base = output_base * TOPK;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = 0.0f;
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += input[input_base + k * d + idx];
    }
    out[output_base + idx] = x;
  }
}

template <int TOPK>
__device__ void moe_sum_vec4_body(float* __restrict__ out, const float* __restrict__ input, int d)
{
  int d4 = d >> 2;
  int output_base4 = (static_cast<int64_t>(blockIdx.x) * d) >> 2;
  int input_base4 = (static_cast<int64_t>(blockIdx.x) * d * TOPK) >> 2;

  const float4* input4 = reinterpret_cast<const float4*>(input);
  float4* out4 = reinterpret_cast<float4*>(out);

  for (int idx = threadIdx.x; idx < d4; idx += blockDim.x) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      float4 v = input4[input_base4 + k * d4 + idx];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
    out4[output_base4 + idx] = acc;
  }
}

extern "C" __global__ void moe_sum_kernel_topk2(float* out, const float* input, int d)
{
  moe_sum_body<2>(out, input, d);
}

extern "C" __global__ void moe_sum_kernel_topk3(float* out, const float* input, int d)
{
  moe_sum_body<3>(out, input, d);
}

extern "C" __global__ void moe_sum_kernel_topk4(float* out, const float* input, int d)
{
  moe_sum_body<4>(out, input, d);
}

extern "C" __global__ void moe_sum_kernel_vec4_topk2(float* out, const float* input, int d)
{
  moe_sum_vec4_body<2>(out, input, d);
}

extern "C" __global__ void moe_sum_kernel_vec4_topk3(float* out, const float* input, int d)
{
  moe_sum_vec4_body<3>(out, input, d);
}

extern "C" __global__ void moe_sum_kernel_vec4_topk4(float* out, const float* input, int d)
{
  moe_sum_vec4_body<4>(out, input, d);
}
