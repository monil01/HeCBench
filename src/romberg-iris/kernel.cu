#include <math.h>

#define ROW_SIZE 17
#define MAX_WORK_GROUP_SIZE 256

__host__ __device__ inline double f(double x)
{
  return exp(x) * sin(x);
}

__device__ inline unsigned int getFirstSetBitPos(int n)
{
  return log2((float)(n & -n)) + 1;
}

extern "C" __global__ void romberg(double a, double b, double* result)
{
  __shared__ double smem[MAX_WORK_GROUP_SIZE * ROW_SIZE];
  if (blockDim.x > MAX_WORK_GROUP_SIZE) return;

  double diff = (b - a) / gridDim.x;
  double step;
  int k;
  int max_eval = (1 << (ROW_SIZE - 1));
  b = a + (blockIdx.x + 1) * diff;
  a += blockIdx.x * diff;

  step = (b - a) / max_eval;

  double local_col[ROW_SIZE];
  for (int i = 0; i < ROW_SIZE; i++) local_col[i] = 0.0;
  if (!threadIdx.x) {
    k = blockDim.x;
    local_col[0] = f(a) + f(b);
  } else {
    k = threadIdx.x;
  }

  for (; k < max_eval; k += blockDim.x) {
    local_col[ROW_SIZE - getFirstSetBitPos(k)] += 2.0 * f(a + step * k);
  }

  for (int i = 0; i < ROW_SIZE; i++) {
    smem[ROW_SIZE * threadIdx.x + i] = local_col[i];
  }
  __syncthreads();

  if (threadIdx.x < ROW_SIZE) {
    double sum = 0.0;
    for (int i = threadIdx.x; i < blockDim.x * ROW_SIZE; i += ROW_SIZE) {
      sum += smem[i];
    }
    smem[threadIdx.x] = sum;
  }

  if (!threadIdx.x) {
    double* table = local_col;
    table[0] = smem[0];

    for (int k = 1; k < ROW_SIZE; k++) {
      table[k] = table[k - 1] + smem[k];
    }

    for (int k = 0; k < ROW_SIZE; k++) {
      table[k] *= (b - a) / (1 << (k + 1));
    }

    for (int col = 0; col < ROW_SIZE - 1; col++) {
      for (int row = ROW_SIZE - 1; row > col; row--) {
        table[row] = table[row] + (table[row] - table[row - 1]) / ((1 << (2 * col + 1)) - 1);
      }
    }

    result[blockIdx.x] = table[ROW_SIZE - 1];
  }
}
