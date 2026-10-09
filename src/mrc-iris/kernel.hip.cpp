#include <hip/hip_runtime.h>

extern "C" __global__ void MRCGradient(
    const int N, const int* Y, const float* X1, const float* X2,
    const float* dOutput, const float margin, float* __restrict__ dX1,
    float* __restrict__ dX2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float dist = -Y[i] * (X1[i] - X2[i]) + margin;
    if (dist < 0.0f) {
      dX1[i] = 0.0f;
      dX2[i] = 0.0f;
    } else {
      dX1[i] = -Y[i] * dOutput[i];
      dX2[i] = Y[i] * dOutput[i];
    }
  }
}

extern "C" __global__ void MRCGradient2(
    const int N, const int* Y, const float* X1, const float* X2,
    const float* dOutput, const float margin, float* __restrict__ dX1,
    float* __restrict__ dX2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float y = Y[i];
    float o = dOutput[i];
    float dist = -y * (X1[i] - X2[i]) + margin;
    dX1[i] = dist < 0.0f ? 0.0f : -y * o;
    dX2[i] = dist < 0.0f ? 0.0f : y * o;
  }
}

extern "C" __global__ void MRCGradient3(
    const int N, const int* Y, const float* X1, const float* X2,
    const float* dOutput, const float margin, float* __restrict__ dX1,
    float* __restrict__ dX2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int vec_count = N / 4;

  if (i < vec_count) {
    int4 y4 = reinterpret_cast<const int4*>(Y)[i];
    float4 x1 = reinterpret_cast<const float4*>(X1)[i];
    float4 x2 = reinterpret_cast<const float4*>(X2)[i];
    float4 do4 = reinterpret_cast<const float4*>(dOutput)[i];

    float4 yf, dist;
    float4 out_x1, out_x2;
    yf.x = y4.x;
    yf.y = y4.y;
    yf.z = y4.z;
    yf.w = y4.w;
    dist.x = fmaf(-yf.x, x1.x - x2.x, margin);
    dist.y = fmaf(-yf.y, x1.y - x2.y, margin);
    dist.z = fmaf(-yf.z, x1.z - x2.z, margin);
    dist.w = fmaf(-yf.w, x1.w - x2.w, margin);

    out_x1.x = dist.x < 0.0f ? 0.0f : -yf.x * do4.x;
    out_x2.x = dist.x < 0.0f ? 0.0f : yf.x * do4.x;
    out_x1.y = dist.y < 0.0f ? 0.0f : -yf.y * do4.y;
    out_x2.y = dist.y < 0.0f ? 0.0f : yf.y * do4.y;
    out_x1.z = dist.z < 0.0f ? 0.0f : -yf.z * do4.z;
    out_x2.z = dist.z < 0.0f ? 0.0f : yf.z * do4.z;
    out_x1.w = dist.w < 0.0f ? 0.0f : -yf.w * do4.w;
    out_x2.w = dist.w < 0.0f ? 0.0f : yf.w * do4.w;

    reinterpret_cast<float4*>(dX1)[i] = out_x1;
    reinterpret_cast<float4*>(dX2)[i] = out_x2;
  }

  for (int j = 4 * vec_count + i; j < N; j += blockDim.x) {
    float yf = static_cast<float>(Y[j]);
    float o = dOutput[j];
    float dist = -yf * (X1[j] - X2[j]) + margin;
    dX1[j] = dist < 0.0f ? 0.0f : -yf * o;
    dX2[j] = dist < 0.0f ? 0.0f : yf * o;
  }
}
