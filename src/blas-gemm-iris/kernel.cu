#define TILE_X 16
#define TILE_Y 16

extern "C" __global__ void matrix_mul_float(const float* a, const float* b, float* c,
                                             int M, int K, int N, float alpha, float beta)
{
  const int row = blockIdx.y * TILE_Y + threadIdx.y;
  const int col = blockIdx.x * TILE_X + threadIdx.x;
  if (row < M && col < N) {
    float s = 0.0f;
    for (int k = 0; k < K; k++) {
      s += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = alpha * s + beta * c[row * N + col];
  }
}

extern "C" __global__ void matrix_mul_double(const double* a, const double* b, double* c,
                                              int M, int K, int N, double alpha, double beta)
{
  const int row = blockIdx.y * TILE_Y + threadIdx.y;
  const int col = blockIdx.x * TILE_X + threadIdx.x;
  if (row < M && col < N) {
    double s = 0.0;
    for (int k = 0; k < K; k++) {
      s += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = alpha * s + beta * c[row * N + col];
  }
}
