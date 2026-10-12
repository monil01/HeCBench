#include "sobel.h"

static inline uchar clamp(float f)
{
  if (f < 0.f) return 0;
  if (f > 255.f) return 255;
  return static_cast<uchar>(f);
}

static void conv(float4& g, const int k[][3], const uchar4* ptr, int c, int width)
{
  g.x = k[0][0] * ptr[c - 1 - width].x + k[0][1] * ptr[c - width].x +
        k[0][2] * ptr[c + 1 - width].x + k[1][0] * ptr[c - 1].x +
        k[1][1] * ptr[c].x + k[1][2] * ptr[c + 1].x +
        k[2][0] * ptr[c - 1 + width].x + k[2][1] * ptr[c + width].x +
        k[2][2] * ptr[c + 1 + width].x;
  g.y = k[0][0] * ptr[c - 1 - width].y + k[0][1] * ptr[c - width].y +
        k[0][2] * ptr[c + 1 - width].y + k[1][0] * ptr[c - 1].y +
        k[1][1] * ptr[c].y + k[1][2] * ptr[c + 1].y +
        k[2][0] * ptr[c - 1 + width].y + k[2][1] * ptr[c + width].y +
        k[2][2] * ptr[c + 1 + width].y;
  g.z = k[0][0] * ptr[c - 1 - width].z + k[0][1] * ptr[c - width].z +
        k[0][2] * ptr[c + 1 - width].z + k[1][0] * ptr[c - 1].z +
        k[1][1] * ptr[c].z + k[1][2] * ptr[c + 1].z +
        k[2][0] * ptr[c - 1 + width].z + k[2][1] * ptr[c + width].z +
        k[2][2] * ptr[c + 1 + width].z;
  g.w = k[0][0] * ptr[c - 1 - width].w + k[0][1] * ptr[c - width].w +
        k[0][2] * ptr[c + 1 - width].w + k[1][0] * ptr[c - 1].w +
        k[1][1] * ptr[c].w + k[1][2] * ptr[c + 1].w +
        k[2][0] * ptr[c - 1 + width].w + k[2][1] * ptr[c + width].w +
        k[2][2] * ptr[c + 1 + width].w;
}

static void magnitude(uchar4& result, float4& gx, float4& gy)
{
  result.x = clamp(std::sqrt(std::pow(gx.x, 2.f) + std::pow(gy.x, 2.f)) / 2.f);
  result.y = clamp(std::sqrt(std::pow(gx.y, 2.f) + std::pow(gy.y, 2.f)) / 2.f);
  result.z = clamp(std::sqrt(std::pow(gx.z, 2.f) + std::pow(gy.z, 2.f)) / 2.f);
  result.w = clamp(std::sqrt(std::pow(gx.w, 2.f) + std::pow(gy.w, 2.f)) / 2.f);
}

void reference(uchar4* verificationOutput, const uchar4* inputImageData, uint width, uint height,
               int)
{
  const int kx[][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
  const int ky[][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
      if (x >= 1 && x < (width - 1) && y >= 1 && y < height - 1) {
        int c = x + y * width;
        float4 gx;
        float4 gy;
        conv(gx, kx, inputImageData, c, width);
        conv(gy, ky, inputImageData, c, width);
        uchar4 result;
        magnitude(result, gx, gy);
        verificationOutput[c] = result;
      }
    }
  }
}
