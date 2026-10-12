#include <hip/hip_runtime.h>

#define X_SIZE 512
#define Y_SIZE 512
#define PI 3.14159265359f
#define WHITE (short)(1)

extern "C" __global__ void affine(const unsigned short* __restrict__ src,
                                  unsigned short* __restrict__ dst)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= X_SIZE || y >= Y_SIZE) return;

  const float lx_rot = 30.0f;
  const float ly_rot = 0.0f;
  const float lx_expan = 0.5f;
  const float ly_expan = 0.5f;
  const int lx_move = 0;
  const int ly_move = 0;
  float affine_m[2][2];
  float i_affine[2][2];
  float beta[2];
  float i_beta[2];

  affine_m[0][0] = lx_expan * cosf(lx_rot * PI / 180.0f);
  affine_m[0][1] = ly_expan * sinf(ly_rot * PI / 180.0f);
  affine_m[1][0] = lx_expan * sinf(lx_rot * PI / 180.0f);
  affine_m[1][1] = ly_expan * cosf(ly_rot * PI / 180.0f);
  beta[0] = lx_move;
  beta[1] = ly_move;

  const float det = (affine_m[0][0] * affine_m[1][1]) - (affine_m[0][1] * affine_m[1][0]);
  if (det == 0.0f) {
    i_affine[0][0] = 1.0f;
    i_affine[0][1] = 0.0f;
    i_affine[1][0] = 0.0f;
    i_affine[1][1] = 1.0f;
    i_beta[0] = -beta[0];
    i_beta[1] = -beta[1];
  } else {
    i_affine[0][0] = affine_m[1][1] / det;
    i_affine[0][1] = -affine_m[0][1] / det;
    i_affine[1][0] = -affine_m[1][0] / det;
    i_affine[1][1] = affine_m[0][0] / det;
    i_beta[0] = -i_affine[0][0] * beta[0] - i_affine[0][1] * beta[1];
    i_beta[1] = -i_affine[1][0] * beta[0] - i_affine[1][1] * beta[1];
  }

  const float x_new = i_beta[0] + i_affine[0][0] * (x - X_SIZE / 2.0f) +
                      i_affine[0][1] * (y - Y_SIZE / 2.0f) + X_SIZE / 2.0f;
  const float y_new = i_beta[1] + i_affine[1][0] * (x - X_SIZE / 2.0f) +
                      i_affine[1][1] * (y - Y_SIZE / 2.0f) + Y_SIZE / 2.0f;

  const int m = (int)floorf(x_new);
  const int n = (int)floorf(y_new);
  const float x_frac = x_new - m;
  const float y_frac = y_new - n;

  unsigned short output_buffer;
  if ((m >= 0) && (m + 1 < X_SIZE) && (n >= 0) && (n + 1 < Y_SIZE)) {
    const float gray_new =
        (1.0f - y_frac) * ((1.0f - x_frac) * (src[(n * X_SIZE) + m]) +
                           x_frac * (src[(n * X_SIZE) + m + 1])) +
        y_frac * ((1.0f - x_frac) * (src[((n + 1) * X_SIZE) + m]) +
                  x_frac * (src[((n + 1) * X_SIZE) + m + 1]));
    output_buffer = (unsigned short)gray_new;
  } else if (((m + 1 == X_SIZE) && (n >= 0) && (n < Y_SIZE)) ||
             ((n + 1 == Y_SIZE) && (m >= 0) && (m < X_SIZE))) {
    output_buffer = src[(n * X_SIZE) + m];
  } else {
    output_buffer = WHITE;
  }

  dst[(y * X_SIZE) + x] = output_buffer;
}
