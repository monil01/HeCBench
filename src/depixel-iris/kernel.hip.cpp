#include <hip/hip_runtime.h>
#include <stdlib.h>

__device__ __forceinline__ unsigned int rgbToyuv(float3 rgba)
{
  float3 yuv;
  yuv.x = 0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z;
  yuv.y = 0.713f * (rgba.x - yuv.x) + 0.5f;
  yuv.z = 0.564f * (rgba.z - yuv.x) + 0.5f;
  yuv.x = __saturatef(yuv.x);
  yuv.y = __saturatef(yuv.y);
  yuv.z = __saturatef(yuv.z);
  return (static_cast<unsigned int>(255) << 24) |
         (static_cast<unsigned int>(yuv.z * 255.f) << 16) |
         (static_cast<unsigned int>(yuv.y * 255.f) << 8) |
         static_cast<unsigned int>(yuv.x * 255.f);
}

__device__ __forceinline__ bool isConnected(unsigned int lnode, unsigned int rnode)
{
  int ly = lnode & 0xff;
  int lu = ((lnode >> 8) & 0xff);
  int lv = ((lnode >> 16) & 0xff);
  int ry = rnode & 0xff;
  int ru = ((rnode >> 8) & 0xff);
  int rv = ((rnode >> 16) & 0xff);
  return !((abs(ly - ry) > 48) || (abs(lu - ru) > 7) || (abs(lv - rv) > 6));
}

__device__ __forceinline__ unsigned int bitCount(unsigned int v)
{
  unsigned int c;
  for (c = 0; v; ++c) v &= v - 1;
  return c;
}

extern "C" __global__ void check_connect(const float3* __restrict__ rgba,
                                          unsigned int* __restrict__ connect,
                                          const int w,
                                          const int h)
{
  unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  int row = center / w;
  int column = center % w;
  int neibor_row, neibor_column;
  unsigned char con = 0;
  unsigned int yuv_c = rgbToyuv(rgba[center]);

  neibor_row = (row > 0 && column > 0) ? (row - 1) : row;
  neibor_column = (column > 0 && row > 0) ? (column - 1) : column;
  unsigned int yuv_ul = rgbToyuv(rgba[neibor_row * w + neibor_column]);
  con += (unsigned int)(!((row == neibor_row) && (column == neibor_column)) && isConnected(yuv_c, yuv_ul));

  neibor_row = (row > 0) ? (row - 1) : row;
  neibor_column = column;
  unsigned int yuv_up = rgbToyuv(rgba[neibor_row * w + neibor_column]);
  con += (unsigned int)(!((row == neibor_row) && (column == neibor_column)) && isConnected(yuv_c, yuv_up)) << 1;

  neibor_row = (row > 0 && column < (w - 1)) ? (row - 1) : row;
  neibor_column = (column < (w - 1) && row > 0) ? (column + 1) : column;
  unsigned int yuv_ur = rgbToyuv(rgba[neibor_row * w + neibor_column]);
  con += (unsigned int)(!((row == neibor_row) && (column == neibor_column)) && isConnected(yuv_c, yuv_ur)) << 2;

  neibor_row = row;
  neibor_column = (column < (w - 1)) ? (column + 1) : column;
  unsigned int yuv_rt = rgbToyuv(rgba[neibor_row * w + neibor_column]);
  con += (unsigned int)(!((row == neibor_row) && (column == neibor_column)) && isConnected(yuv_c, yuv_rt)) << 3;

  neibor_row = (row < (h - 1) && column < (w - 1)) ? (row + 1) : row;
  neibor_column = (column < (w - 1) && row < (h - 1)) ? (column + 1) : column;
  unsigned int yuv_lr = rgbToyuv(rgba[neibor_row * w + neibor_column]);
  con += (unsigned int)(!((row == neibor_row) && (column == neibor_column)) && isConnected(yuv_c, yuv_lr)) << 4;

  neibor_row = (row < (h - 1)) ? (row + 1) : row;
  neibor_column = column;
  unsigned int yuv_lw = rgbToyuv(rgba[neibor_row * w + neibor_column]);
  con += (unsigned int)(!((row == neibor_row) && (column == neibor_column)) && isConnected(yuv_c, yuv_lw)) << 5;

  neibor_row = (row < (h - 1) && column > 0) ? (row + 1) : row;
  neibor_column = (column > 0 && row < (h - 1)) ? (column - 1) : column;
  unsigned int yuv_ll = rgbToyuv(rgba[neibor_row * w + neibor_column]);
  con += (unsigned int)(!((row == neibor_row) && (column == neibor_column)) && isConnected(yuv_c, yuv_ll)) << 6;

  neibor_row = row;
  neibor_column = (column > 0) ? (column - 1) : column;
  unsigned int yuv_lt = rgbToyuv(rgba[neibor_row * w + neibor_column]);
  con += (unsigned int)(!((row == neibor_row) && (column == neibor_column)) && isConnected(yuv_c, yuv_lt)) << 7;

  connect[center] = (yuv_c >> 16 & 0xFF) << 24 | (yuv_c >> 8 & 0xFF) << 16 |
                    (yuv_c & 0xFF) << 8 | con;
}

extern "C" __global__ void eliminate_crosses(const unsigned int* __restrict__ id,
                                              unsigned int* __restrict__ od,
                                              const int w,
                                              const int h)
{
  unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  int row = center / w;
  int column = center % w;
  int start_row = (row > 2) ? row - 3 : 0;
  int start_column = (column > 2) ? column - 3 : 0;
  int end_row = (row < w - 4) ? row + 4 : w - 1;
  int end_column = (column < h - 4) ? column + 4 : h - 1;
  int weight_l = 0;
  int weight_r = 0;
  od[center] = 0;
  if ((row < h - 1) && (column < w - 1)) {
    od[center] = (id[center] & 0x08) >> 3 |
                 (((id[center + w + 1] & 0x02) >> 1) << 1) |
                 (((id[center + w + 1] & 0x80) >> 7) << 2) |
                 (((id[center] & 0x20) >> 5) << 3);

    if ((id[center] & 0x10 && id[center + 1] & 0x40)) {
      if (id[center] & 0x28 && id[center + 1] & 0xA0) {
        od[center] = ((id[center] >> 8) & 0xFFFFFF) << 8 | od[center];
        return;
      }

      if (id[center] == 0x10) weight_l += 5;
      if (id[center + 1] == 0x40) weight_r += 5;

      int sum_l = 0;
      int sum_r = 0;
      for (int i = start_row; i <= end_row; ++i) {
        for (int j = start_column; j <= end_column; ++j) {
          if (i * w + j != center && i * w + j != center + 1) {
            sum_l += isConnected(id[center] >> 8, id[i * w + j] >> 8);
            sum_r += isConnected(id[center + 1] >> 8, id[i * w + j] >> 8);
          }
        }
      }

      weight_r += (sum_l > sum_r) ? (sum_l - sum_r) : 0;
      weight_l += (sum_l < sum_r) ? (sum_r - sum_l) : 0;

      int c_row = row;
      int c_column = column;
      unsigned int curve_l = id[c_row * w + c_column] & 0xFF;
      unsigned int edge_l = 16;
      sum_l = 1;
      while (bitCount(curve_l) == 2 && sum_l < w * h) {
        edge_l = curve_l - edge_l;
        switch (edge_l) {
          case 1: c_row -= 1; c_column -= 1; break;
          case 2: c_row -= 1; break;
          case 4: c_row -= 1; c_column += 1; break;
          case 8: c_column += 1; break;
          case 16: c_row += 1; c_column += 1; break;
          case 32: c_row += 1; break;
          case 64: c_row += 1; c_column -= 1; break;
          case 128: c_column -= 1; break;
        }
        edge_l = (edge_l > 8) ? edge_l >> 4 : edge_l << 4;
        curve_l = id[c_row * w + c_column] & 0xFF;
        ++sum_l;
      }
      c_row = row + 1;
      c_column = column + 1;
      curve_l = id[c_row * w + c_column] & 0xFF;
      edge_l = 1;
      while (bitCount(curve_l) == 2 && sum_l < w * h) {
        edge_l = curve_l - edge_l;
        switch (edge_l) {
          case 1: c_row -= 1; c_column -= 1; break;
          case 16: c_row += 1; c_column += 1; break;
          case 2: c_row -= 1; break;
          case 4: c_row -= 1; c_column += 1; break;
          case 8: c_column += 1; break;
          case 32: c_row += 1; break;
          case 64: c_row += 1; c_column -= 1; break;
          case 128: c_column -= 1; break;
        }
        edge_l = (edge_l > 8) ? edge_l >> 4 : edge_l << 4;
        curve_l = id[c_row * w + c_column] & 0xFF;
        ++sum_l;
      }
      c_row = row;
      c_column = column + 1;
      unsigned int curve_r = id[c_row * w + c_column] & 0xFF;
      unsigned int edge_r = 64;
      sum_r = 1;
      while (bitCount(curve_r) == 2 && sum_r < w * h) {
        edge_r = curve_r - edge_r;
        switch (edge_r) {
          case 64: c_row += 1; c_column -= 1;
          case 1: c_row -= 1; c_column -= 1; break;
          case 2: c_row -= 1; break;
          case 4: c_row -= 1; c_column += 1; break;
          case 8: c_column += 1; break;
          case 32: c_row += 1; break;
          case 16: c_row += 1; c_column += 1; break;
          case 128: c_column -= 1; break;
        }
        edge_r = (edge_r > 8) ? edge_r >> 4 : edge_r << 4;
        curve_r = id[c_row * w + c_column] & 0xFF;
        ++sum_r;
      }
      c_row = row + 1;
      c_column = column;
      curve_r = id[c_row * w + c_column] & 0xFF;
      edge_r = 4;
      while (bitCount(curve_r) == 2 && sum_r < w * h) {
        edge_r = curve_r - edge_r;
        switch (edge_r) {
          case 4: c_row -= 1; c_column += 1; break;
          case 16: c_row += 1; c_column += 1; break;
          case 2: c_row -= 1; break;
          case 1: c_row -= 1; c_column -= 1; break;
          case 8: c_column += 1; break;
          case 32: c_row += 1; break;
          case 64: c_row += 1; c_column -= 1; break;
          case 128: c_column -= 1; break;
        }
        edge_r = (edge_r > 8) ? edge_r >> 4 : edge_r << 4;
        curve_r = id[c_row * w + c_column] & 0xFF;
        ++sum_r;
      }

      weight_l += (sum_l > sum_r) ? (sum_l - sum_r) : 0;
      weight_r += (sum_l < sum_r) ? (sum_r - sum_l) : 0;

      if (weight_l > weight_r) {
        od[center] |= 0x10;
        od[center] = ((id[center] >> 8) & 0xFFFFFF) << 8 | od[center];
        return;
      } else {
        if (weight_r > weight_l) {
          od[center] |= 0x20;
          od[center] = ((id[center] >> 8) & 0xFFFFFF) << 8 | od[center];
          return;
        }
      }
    }
    od[center] = od[center] | (((id[center] & 0x10) >> 4) << 4) |
                 (((id[center + 1] & 0x40) >> 6) << 5);
  }
  od[center] = ((id[center] >> 8) & 0xFFFFFF) << 8 | od[center];
}
