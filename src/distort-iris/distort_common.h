#ifndef DISTORT_IRIS_COMMON_H
#define DISTORT_IRIS_COMMON_H

struct Pixel {
  unsigned char x;
  unsigned char y;
  unsigned char z;
};

struct Properties {
  float K;
  float centerX;
  float centerY;
  int width;
  int height;
  float thresh;
  float xscale;
  float yscale;
  float xshift;
  float yshift;
};

#endif
