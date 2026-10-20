#include <cmath>
#include <cstdio>
#include <initializer_list>
inline float Tex2D(const float *t, int w, int h, int s, float x, float y) {
  // integer parts in floating point format
  float intPartX, intPartY;

  // get fractional parts of coordinates
  float dx = fabsf(modff(x, &intPartX));
  float dy = fabsf(modff(y, &intPartY));

  // assume pixels are squares
  // one of the corners
  int ix0 = (int)intPartX;
  int iy0 = (int)intPartY;

  // mirror out-of-range position
  if (ix0 < 0) ix0 = abs(ix0 + 1);

  if (iy0 < 0) iy0 = abs(iy0 + 1);

  if (ix0 >= w) ix0 = w * 2 - ix0 - 1;

  if (iy0 >= h) iy0 = h * 2 - iy0 - 1;

  // corner which is opposite to (ix0, iy0)
  int ix1 = ix0 + 1;
  int iy1 = iy0 + 1;

  if (ix1 >= w) ix1 = w * 2 - ix1 - 1;

  if (iy1 >= h) iy1 = h * 2 - iy1 - 1;

  float res = t[ix0 + iy0 * s] * (1.0f - dx) * (1.0f - dy);
  res += t[ix1 + iy0 * s] * dx * (1.0f - dy);
  res += t[ix0 + iy1 * s] * (1.0f - dx) * dy;
  res += t[ix1 + iy1 * s] * dx * dy;

  return res;
}
int main(){float t[]={0,1,2};for(float x:{-.25f,3.25f})printf("x=%.8g source_CPU=%.8g\n",x,Tex2D(t,3,1,3,x,0));}
