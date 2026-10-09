#include <cuda_fp16.h>

extern "C" __global__ void gelu_bias_loop(__half* src, const __half* bias, int width, int height)
{
  int x = blockIdx.x;
  int batch = blockIdx.y;

  int index = batch * width * height + x * width;
  for (int y = threadIdx.x; y < width; y = y + blockDim.x) {
    auto v_bias = bias[y];
    auto v_src = src[index + y];
    auto v = v_src + v_bias;
    auto t = __half2float(v);
    t = (0.5f * t * (1.0f + tanhf(0.79788456f * (t + 0.044715f * t * t * t))));
    src[index + y] = __float2half_rn(t);
  }
}

extern "C" __global__ void gelu_bias_loop_base(__half* src, const __half* bias,
                                               int width, int height)
{
  int x = blockIdx.x;
  int batch = blockIdx.y;

  int index = batch * width * height + x * width;
  for (int y = threadIdx.x; y < width; y = y + blockDim.x) {
    auto v_bias = bias[y];
    auto v_src = src[index + y];
    auto v = v_src + v_bias;
    auto t = __half2float(v);
    t = (0.5f * t * (1.0f + tanhf(0.79788456f * (t + 0.044715f * t * t * t))));
    src[index + y] = __float2half_rn(t);
  }
}
