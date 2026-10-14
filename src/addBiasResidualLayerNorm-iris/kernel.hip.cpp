#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "kernels.h"
extern "C" __global__ void addBiasResidualLayerNorm_float_v1(float* out, const float* input, const float* bias, const float* gamma, const float* beta, float epsilon, int n) {
  if (warpSize == 32) {
    if (n == 256) addBiasResidualPostLayerNorm<float, 1, 32>(out,input,bias,gamma,beta,epsilon,n);
    else if (n == 512) addBiasResidualPostLayerNorm<float, 2, 32>(out,input,bias,gamma,beta,epsilon,n);
    else generalAddBiasResidualPostLayerNorm<float, 32>(out,input,bias,gamma,beta,epsilon,n);
  }
  else if (warpSize == 64) {
    if (n == 256) addBiasResidualPostLayerNorm<float, 1, 64>(out,input,bias,gamma,beta,epsilon,n);
    else if (n == 512) addBiasResidualPostLayerNorm<float, 2, 64>(out,input,bias,gamma,beta,epsilon,n);
    else generalAddBiasResidualPostLayerNorm<float, 64>(out,input,bias,gamma,beta,epsilon,n);
  }
}
extern "C" __global__ void addBiasResidualLayerNorm_float_v2(float* out, const float* input, const float* bias, const float* gamma, const float* beta, float epsilon, int n) {
  if (warpSize == 32) {
    if (n / 8 >= 32) addBiasResidualPostLayerNormV2<float, 32>(out,input,bias,gamma,beta,epsilon,n);
    else addBiasResidualPostLayerNormV2<float, 32>(out,input,bias,gamma,beta,epsilon,n);
  }
  else if (warpSize == 64) {
    if (n / 8 >= 64) addBiasResidualPostLayerNormV2<float, 64>(out,input,bias,gamma,beta,epsilon,n);
    else addBiasResidualPostLayerNormV2<float, 32>(out,input,bias,gamma,beta,epsilon,n);
  }
}
extern "C" __global__ void addBiasResidualLayerNorm_half_v1(__half* out, const __half* input, const __half* bias, const __half* gamma, const __half* beta, float epsilon, int n) {
  if (warpSize == 32) {
    if (n == 256) addBiasResidualPostLayerNorm<__half, 1, 32>(out,input,bias,gamma,beta,epsilon,n);
    else if (n == 512) addBiasResidualPostLayerNorm<__half, 2, 32>(out,input,bias,gamma,beta,epsilon,n);
    else generalAddBiasResidualPostLayerNorm<__half, 32>(out,input,bias,gamma,beta,epsilon,n);
  }
  else if (warpSize == 64) {
    if (n == 256) addBiasResidualPostLayerNorm<__half, 1, 64>(out,input,bias,gamma,beta,epsilon,n);
    else if (n == 512) addBiasResidualPostLayerNorm<__half, 2, 64>(out,input,bias,gamma,beta,epsilon,n);
    else generalAddBiasResidualPostLayerNorm<__half, 64>(out,input,bias,gamma,beta,epsilon,n);
  }
}
extern "C" __global__ void addBiasResidualLayerNorm_half_v2(__half* out, const __half* input, const __half* bias, const __half* gamma, const __half* beta, float epsilon, int n) {
  if (warpSize == 32) {
    if (n / 8 >= 32) addBiasResidualPostLayerNormV2<__half, 32>(out,input,bias,gamma,beta,epsilon,n);
    else addBiasResidualPostLayerNormV2<__half, 32>(out,input,bias,gamma,beta,epsilon,n);
  }
  else if (warpSize == 64) {
    if (n / 8 >= 64) addBiasResidualPostLayerNormV2<__half, 64>(out,input,bias,gamma,beta,epsilon,n);
    else addBiasResidualPostLayerNormV2<__half, 32>(out,input,bias,gamma,beta,epsilon,n);
  }
}
extern "C" __global__ void addBiasResidualLayerNorm_bfloat16_v1(__hip_bfloat16* out, const __hip_bfloat16* input, const __hip_bfloat16* bias, const __hip_bfloat16* gamma, const __hip_bfloat16* beta, float epsilon, int n) {
  if (warpSize == 32) {
    if (n == 256) addBiasResidualPostLayerNorm<__hip_bfloat16, 1, 32>(out,input,bias,gamma,beta,epsilon,n);
    else if (n == 512) addBiasResidualPostLayerNorm<__hip_bfloat16, 2, 32>(out,input,bias,gamma,beta,epsilon,n);
    else generalAddBiasResidualPostLayerNorm<__hip_bfloat16, 32>(out,input,bias,gamma,beta,epsilon,n);
  }
  else if (warpSize == 64) {
    if (n == 256) addBiasResidualPostLayerNorm<__hip_bfloat16, 1, 64>(out,input,bias,gamma,beta,epsilon,n);
    else if (n == 512) addBiasResidualPostLayerNorm<__hip_bfloat16, 2, 64>(out,input,bias,gamma,beta,epsilon,n);
    else generalAddBiasResidualPostLayerNorm<__hip_bfloat16, 64>(out,input,bias,gamma,beta,epsilon,n);
  }
}
extern "C" __global__ void addBiasResidualLayerNorm_bfloat16_v2(__hip_bfloat16* out, const __hip_bfloat16* input, const __hip_bfloat16* bias, const __hip_bfloat16* gamma, const __hip_bfloat16* beta, float epsilon, int n) {
  if (warpSize == 32) {
    if (n / 8 >= 32) addBiasResidualPostLayerNormV2<__hip_bfloat16, 32>(out,input,bias,gamma,beta,epsilon,n);
    else addBiasResidualPostLayerNormV2<__hip_bfloat16, 32>(out,input,bias,gamma,beta,epsilon,n);
  }
  else if (warpSize == 64) {
    if (n / 8 >= 64) addBiasResidualPostLayerNormV2<__hip_bfloat16, 64>(out,input,bias,gamma,beta,epsilon,n);
    else addBiasResidualPostLayerNormV2<__hip_bfloat16, 32>(out,input,bias,gamma,beta,epsilon,n);
  }
}
