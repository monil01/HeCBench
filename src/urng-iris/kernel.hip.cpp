#include <hip/hip_runtime.h>

#define GROUP_SIZE 256
#define FACTOR 25
#define IA 16807
#define IM 2147483647
#define AM (1.f / (float)IM)
#define IQ 127773
#define IR 2836
#define NTAB 16
#define NDIV (1 + (IM - 1) / NTAB)

__device__ __forceinline__ float4 convert_float4(uchar4 v)
{
  return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}

__device__ __forceinline__ uchar4 convert_uchar4_sat(float4 v)
{
  uchar4 res;
  res.x = (unsigned char)((v.x > 255.f) ? 255.f : (v.x < 0.f ? 0.f : v.x));
  res.y = (unsigned char)((v.y > 255.f) ? 255.f : (v.y < 0.f ? 0.f : v.y));
  res.z = (unsigned char)((v.z > 255.f) ? 255.f : (v.z < 0.f ? 0.f : v.z));
  res.w = (unsigned char)((v.w > 255.f) ? 255.f : (v.w < 0.f ? 0.f : v.w));
  return res;
}

__device__ float ran1(int idum, int* iv)
{
  int iy = 0;
  const int tid = threadIdx.x;
  for (int j = NTAB; j >= 0; j--) {
    const int k = idum / IQ;
    idum = IA * (idum - k * IQ) - IR * k;
    if (idum < 0) idum += IM;
    if (j < NTAB) iv[NTAB * j + tid] = idum;
  }
  iy = iv[tid];

  const int k = idum / IQ;
  idum = IA * (idum - k * IQ) - IR * k;
  if (idum < 0) idum += IM;

  const int j = iy / NDIV;
  iy = iv[NTAB * j + tid];
  return AM * iy;
}

__device__ __forceinline__ float4 operator+(float4 a, float b)
{
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

extern "C" __global__ void noise_uniform(const uchar4* __restrict__ inputImage,
                                         uchar4* __restrict__ outputImage,
                                         const int size,
                                         const int factor)
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= size) return;

  const float4 temp = convert_float4(inputImage[pos]);
  const float avg = (temp.x + temp.y + temp.z + temp.w) / 4.0f;

  __shared__ int iv[NTAB * GROUP_SIZE];
  float dev = ran1(-(int)avg, iv);
  dev = (dev - 0.55f) * (float)factor;
  outputImage[pos] = convert_uchar4_sat(temp + dev);
}
