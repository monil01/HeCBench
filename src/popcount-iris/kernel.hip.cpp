#include <hip/hip_runtime.h>

#define m1  0x5555555555555555UL
#define m2  0x3333333333333333UL
#define m4  0x0f0f0f0f0f0f0f0fUL
#define h01 0x0101010101010101UL

extern "C" __global__ void pc1(const unsigned long* __restrict__ data,
                                int* __restrict__ r,
                                const int length)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= length) return;
  unsigned long x = data[i];
  x -= (x >> 1) & m1;
  x = (x & m2) + ((x >> 2) & m2);
  x = (x + (x >> 4)) & m4;
  x += x >> 8;
  x += x >> 16;
  x += x >> 32;
  r[i] = x & 0x7f;
}

extern "C" __global__ void pc2(const unsigned long* __restrict__ data,
                                int* __restrict__ r,
                                const int length)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= length) return;
  unsigned long x = data[i];
  x -= (x >> 1) & m1;
  x = (x & m2) + ((x >> 2) & m2);
  x = (x + (x >> 4)) & m4;
  r[i] = (x * h01) >> 56;
}

extern "C" __global__ void pc3(const unsigned long* __restrict__ data,
                                int* __restrict__ r,
                                const int length)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= length) return;
  char count;
  unsigned long x = data[i];
  for (count = 0; x; count++) x &= x - 1;
  r[i] = count;
}

extern "C" __global__ void pc4(const unsigned long* __restrict__ data,
                                int* __restrict__ r,
                                const int length)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= length) return;
  unsigned long x = data[i];
  char cnt = 0;
  for (char bit = 0; bit < 64; bit++) {
    cnt = cnt + (x & 0x1);
    x = x >> 1;
  }
  r[i] = cnt;
}

extern "C" __global__ void pc5(const unsigned long* __restrict__ data,
                                int* __restrict__ r,
                                const int length,
                                const unsigned char* __restrict__ lut)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= length) return;
  unsigned long x = data[i];
  unsigned char i1 = lut[(x & 0xFF)];
  unsigned char i2 = lut[(x >> 8) & 0xFF];
  unsigned char i3 = lut[(x >> 16) & 0xFF];
  unsigned char i4 = lut[(x >> 24) & 0xFF];
  unsigned char i5 = lut[(x >> 32) & 0xFF];
  unsigned char i6 = lut[(x >> 40) & 0xFF];
  unsigned char i7 = lut[(x >> 48) & 0xFF];
  unsigned char i8 = lut[(x >> 56) & 0xFF];
  r[i] = (i1 + i2) + (i3 + i4) + (i5 + i6) + (i7 + i8);
}

extern "C" __global__ void pc6(const unsigned long* __restrict__ data,
                                int* __restrict__ r,
                                const int length)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= length) return;
  r[i] = __popcll(data[i]);
}
