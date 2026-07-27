#ifndef SOBEL_H
#define SOBEL_H

#include <chrono>
#include <cmath>

typedef unsigned char uchar;

// The -cuda variant gets uchar4/float4 from <cuda.h> and the -omp variant
// gets them from an SDKBitMap.h path guarded by _OPENMP. In the -serial
// variant we compile with plain g++, so define these vector types locally
// BEFORE including SDKBitMap.h (which uses them).
#ifndef HECBENCH_HAVE_VECTOR_TYPES
#define HECBENCH_HAVE_VECTOR_TYPES
typedef unsigned int uint;
typedef struct __attribute__((__aligned__(4))) {
  unsigned char x, y, z, w;
} uchar4;
typedef struct __attribute__((__aligned__(16))) {
  float x, y, z, w;
} float4;
#endif

#include "SDKBitMap.h"

void reference (uchar4 *verificationOutput,
                const uchar4 *inputImageData, 
                const uint width,
                const uint height,
                const int pixelSize);

#endif
