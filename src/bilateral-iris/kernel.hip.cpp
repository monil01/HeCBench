#include <hip/hip_runtime.h>
#include <math.h>

template <int R>
__device__ void bilateral_body(const float* __restrict__ in, float* __restrict__ out,
                               int w, int h, float a_square, float variance_I,
                               float variance_spatial)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= w || idy >= h) return;
  int id = idy * w + idx;
  float I = in[id];
  float res = 0.0f;
  float normalization = 0.0f;
  for (int i = -R; i <= R; i++) {
    for (int j = -R; j <= R; j++) {
      int idk = idx + i;
      int idl = idy + j;
      if (idk < 0) idk = -idk;
      if (idl < 0) idl = -idl;
      if (idk > w - 1) idk = w - 1 - i;
      if (idl > h - 1) idl = h - 1 - j;
      int id_w = idl * w + idk;
      float I_w = in[id_w];
      float range = -(I - I_w) * (I - I_w) / (2.0f * variance_I);
      float spatial = -((idk - idx) * (idk - idx) + (idl - idy) * (idl - idy)) /
                      (2.0f * variance_spatial);
      float weight = a_square * expf(spatial + range);
      normalization += weight;
      res += I_w * weight;
    }
  }
  out[id] = res / normalization;
}

extern "C" __global__ void bilateral_filter_r3(const float* in, float* out, int w, int h,
                                                float a_square, float variance_I,
                                                float variance_spatial)
{
  bilateral_body<3>(in, out, w, h, a_square, variance_I, variance_spatial);
}

extern "C" __global__ void bilateral_filter_r6(const float* in, float* out, int w, int h,
                                                float a_square, float variance_I,
                                                float variance_spatial)
{
  bilateral_body<6>(in, out, w, h, a_square, variance_I, variance_spatial);
}

extern "C" __global__ void bilateral_filter_r9(const float* in, float* out, int w, int h,
                                                float a_square, float variance_I,
                                                float variance_spatial)
{
  bilateral_body<9>(in, out, w, h, a_square, variance_I, variance_spatial);
}
