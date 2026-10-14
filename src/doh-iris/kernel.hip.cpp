#include <hip/hip_runtime.h>

using IMAGE_T = float;
using INT_T = int;

__device__ inline int _clip(const int x, const int low, const int high) {
  if (x > high) return high;
  if (x < low) return low;
  return x;
}

__device__ inline IMAGE_T _integ(const IMAGE_T* img, const INT_T img_rows,
                                 const INT_T img_cols, int r, int c,
                                 const int rl, const int cl) {
  r = _clip(r, 0, img_rows - 1);
  c = _clip(c, 0, img_cols - 1);

  const int r2 = _clip(r + rl, 0, img_rows - 1);
  const int c2 = _clip(c + cl, 0, img_cols - 1);

  IMAGE_T ans = img[r * img_cols + c] + img[r2 * img_cols + c2] -
                img[r * img_cols + c2] - img[r2 * img_cols + c];
  return max(static_cast<IMAGE_T>(0), ans);
}

extern "C" __global__ void hessian_matrix_det(
    const IMAGE_T* img, const INT_T img_rows, const INT_T img_cols,
    const IMAGE_T sigma, IMAGE_T* out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= img_rows * img_cols) return;

  const int r = tid / img_cols;
  const int c = tid % img_cols;
  int size = static_cast<int>(static_cast<IMAGE_T>(3.0) * sigma);

  const int b = (size - 1) / 2 + 1;
  const int l = size / 3;
  const int w = size;
  const IMAGE_T w_i = static_cast<IMAGE_T>(1.0) / (size * size);

  const IMAGE_T tl = _integ(img, img_rows, img_cols, r - l, c - l, l, l);
  const IMAGE_T br = _integ(img, img_rows, img_cols, r + 1, c + 1, l, l);
  const IMAGE_T bl = _integ(img, img_rows, img_cols, r - l, c + 1, l, l);
  const IMAGE_T tr = _integ(img, img_rows, img_cols, r + 1, c - l, l, l);

  IMAGE_T dxy = bl + tr - tl - br;
  dxy = -dxy * w_i;

  IMAGE_T mid = _integ(img, img_rows, img_cols, r - l + 1, c - l, 2 * l - 1, w);
  IMAGE_T side = _integ(img, img_rows, img_cols, r - l + 1, c - l / 2, 2 * l - 1, l);

  IMAGE_T dxx = mid - static_cast<IMAGE_T>(3) * side;
  dxx = -dxx * w_i;

  mid = _integ(img, img_rows, img_cols, r - l, c - b + 1, w, 2 * b - 1);
  side = _integ(img, img_rows, img_cols, r - b / 2, c - b + 1, b, 2 * b - 1);

  IMAGE_T dyy = mid - static_cast<IMAGE_T>(3) * side;
  dyy = -dyy * w_i;

  out[tid] = dxx * dyy - static_cast<IMAGE_T>(0.81) * (dxy * dxy);
}
