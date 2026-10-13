#ifndef _SHARED
#define _SHARED

#include <cstdint>
#if !defined(__HIPCC__) && !defined(__CUDACC__)
struct uchar3 { unsigned char x, y, z; };
#endif
static_assert(sizeof(uchar3)==3, "packed RGB layout required");

struct Params {
  uint32_t oWidth;
  uint32_t oHeight;
  uint32_t iWidth;
  uint32_t iHeight;
     float pWidth;
     float pHeight;
     float lambda;
  uint32_t repeat;
};

#endif
