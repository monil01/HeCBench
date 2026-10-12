#ifndef IRIS_SOBEL_H
#define IRIS_SOBEL_H

#include <cmath>

typedef unsigned char uchar;
typedef unsigned int uint;

typedef struct __attribute__((__aligned__(4))) {
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
} uchar4;

typedef struct __attribute__((__aligned__(16))) {
  float x;
  float y;
  float z;
  float w;
} float4;

void reference(uchar4* verificationOutput, const uchar4* inputImageData, uint width, uint height,
               int pixelSize);

#endif
