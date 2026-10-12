#pragma once

#include <cstdint>

using uchar_misaligned = unsigned char;
using ushort_misaligned = unsigned short int;

struct uchar4_misaligned {
  unsigned char r, g, b, a;
};

struct uint2_misaligned {
  unsigned int l, a;
};

struct uint3_misaligned {
  unsigned int r, g, b;
};

struct uint4_misaligned {
  unsigned int r, g, b, a;
};

struct uint8_misaligned {
  uint4_misaligned c1, c2;
};

struct alignas(4) uchar4_aligned {
  unsigned char r, g, b, a;
};

using uint_aligned = unsigned int;

struct alignas(8) uint2_aligned {
  unsigned int l, a;
};

struct alignas(16) uint3_aligned {
  unsigned int r, g, b;
};

struct alignas(16) uint4_aligned {
  unsigned int r, g, b, a;
};

struct alignas(16) uint8_aligned {
  uint4_aligned c1, c2;
};

inline int iDivDown(int a, int b) {
  return a / b;
}

inline int iAlignDown(int a, int b) {
  return a - a % b;
}
