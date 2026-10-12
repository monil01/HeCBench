#include <hip/hip_runtime.h>

typedef unsigned char uchar;

static inline __device__ float4 convert_float4(uchar4 data)
{
  return make_float4(data.x, data.y, data.z, data.w);
}

static inline __device__ uchar4 convert_uchar4(float4 v)
{
  uchar4 res;
  res.x = static_cast<uchar>((v.x > 255.f) ? 255.f : (v.x < 0.f ? 0.f : v.x));
  res.y = static_cast<uchar>((v.y > 255.f) ? 255.f : (v.y < 0.f ? 0.f : v.y));
  res.z = static_cast<uchar>((v.z > 255.f) ? 255.f : (v.z < 0.f ? 0.f : v.z));
  res.w = static_cast<uchar>((v.w > 255.f) ? 255.f : (v.w < 0.f ? 0.f : v.w));
  return res;
}

static inline __device__ float4 operator+(float4 a, float4 b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static inline __device__ float4 operator-(float4 a, float4 b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

static inline __device__ float4 operator*(float4 a, float4 b)
{
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

extern "C" __global__ void sobel_filter(const uchar4* __restrict__ inputImage,
                                         uchar4* __restrict__ outputImage,
                                         const uint width,
                                         const uint height)
{
  uint x = blockDim.x * blockIdx.x + threadIdx.x;
  uint y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height) {
    int c = x + y * width;
    if (!(x >= 1 && x < (width - 1) && y >= 1 && y < height - 1)) {
      outputImage[c] = make_uchar4(0, 0, 0, 0);
      return;
    }
    float4 i00 = convert_float4(inputImage[c - 1 - width]);
    float4 i01 = convert_float4(inputImage[c - width]);
    float4 i02 = convert_float4(inputImage[c + 1 - width]);
    float4 i10 = convert_float4(inputImage[c - 1]);
    float4 i12 = convert_float4(inputImage[c + 1]);
    float4 i20 = convert_float4(inputImage[c - 1 + width]);
    float4 i21 = convert_float4(inputImage[c + width]);
    float4 i22 = convert_float4(inputImage[c + 1 + width]);
    const float4 two = make_float4(2.f, 2.f, 2.f, 2.f);
    float4 gx = i00 + two * i10 + i20 - i02 - two * i12 - i22;
    float4 gy = i00 - i20 + two * i01 - two * i21 + i02 - i22;
    outputImage[c] = convert_uchar4(make_float4(sqrtf(gx.x * gx.x + gy.x * gy.x) / 2.f,
                                                sqrtf(gx.y * gx.y + gy.y * gy.y) / 2.f,
                                                sqrtf(gx.z * gx.z + gy.z * gy.z) / 2.f,
                                                sqrtf(gx.w * gx.w + gy.w * gy.w) / 2.f));
  }
}
