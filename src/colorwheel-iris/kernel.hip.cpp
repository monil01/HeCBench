#include <hip/hip_runtime.h>

#define RY 15
#define YG 6
#define GC 4
#define CB 11
#define BM 13
#define MR 6
#define MAXCOLS (RY + YG + GC + BM + CB + MR)
using uchar = unsigned char;

__device__ void setcols(int cw[MAXCOLS][3], int r, int g, int b, int k) {
  cw[k][0] = r;
  cw[k][1] = g;
  cw[k][2] = b;
}

__device__ void computeColor(float fx, float fy, uchar* pix) {
  int cw[MAXCOLS][3];
  int k = 0;
  for (int i = 0; i < RY; i++) setcols(cw, 255, 255 * i / RY, 0, k++);
  for (int i = 0; i < YG; i++) setcols(cw, 255 - 255 * i / YG, 255, 0, k++);
  for (int i = 0; i < GC; i++) setcols(cw, 0, 255, 255 * i / GC, k++);
  for (int i = 0; i < CB; i++) setcols(cw, 0, 255 - 255 * i / CB, 255, k++);
  for (int i = 0; i < BM; i++) setcols(cw, 255 * i / BM, 0, 255, k++);
  for (int i = 0; i < MR; i++) setcols(cw, 255, 0, 255 - 255 * i / MR, k++);

  float rad = sqrtf(fx * fx + fy * fy);
  float a = atan2f(-fy, -fx) / static_cast<float>(M_PI);
  float fk = (a + 1.f) / 2.f * (MAXCOLS - 1);
  int k0 = static_cast<int>(fk);
  int k1 = (k0 + 1) % MAXCOLS;
  float f = fk - k0;
  for (int b = 0; b < 3; b++) {
    float col0 = cw[k0][b] / 255.f;
    float col1 = cw[k1][b] / 255.f;
    float col = (1.f - f) * col0 + f * col1;
    if (rad <= 1)
      col = 1.f - rad * (1.f - col);
    else
      col *= .75f;
    pix[2 - b] = static_cast<int>(255.f * col);
  }
}

extern "C" __global__ void color(uchar* pix, int size, int half_size,
                                 float range, float truerange) {
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  if (y >= size || x >= size) return;

  size_t idx = (static_cast<size_t>(y) * size + x) * 3;
  if (x == half_size || y == half_size) {
    pix[idx] = 0;
    pix[idx + 1] = 0;
    pix[idx + 2] = 0;
    return;
  }

  float fx = static_cast<float>(x) / static_cast<float>(half_size) * range - range;
  float fy = static_cast<float>(y) / static_cast<float>(half_size) * range - range;
  computeColor(fx / truerange, fy / truerange, pix + idx);
}
