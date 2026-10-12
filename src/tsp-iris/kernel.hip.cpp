#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>

#define tilesize 128
#define dist(a, b) int(sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])))
#define swap(a, b) {float tmp = a;  a = b;  b = tmp;}

__device__ float LCG_random(unsigned int *seed)
{
  const unsigned int m = 2147483648u;
  const unsigned int a = 26757677u;
  const unsigned int c = 1u;
  *seed = (a * (*seed) + c) % m;
  return static_cast<float>(*seed) / static_cast<float>(m);
}

extern "C" __global__ void TwoOpt(
    int cities,
    const float *__restrict posx_d,
    const float *__restrict posy_d,
    int *__restrict glob_d,
    int *__restrict climbs_d,
    int *__restrict best_d)
{
  __shared__ int buf_s[256];
  __shared__ float px_s[tilesize];
  __shared__ float py_s[tilesize];
  __shared__ int bf_s[tilesize];

  int *buf = &glob_d[blockIdx.x * ((3 * cities + 2 + 31) / 32 * 32)];
  float *px = reinterpret_cast<float *>(&buf[cities]);
  float *py = &px[cities + 1];

  for (int i = threadIdx.x; i < cities; i += blockDim.x) px[i] = posx_d[i];
  for (int i = threadIdx.x; i < cities; i += blockDim.x) py[i] = posy_d[i];
  __syncthreads();

  if (threadIdx.x == 0) {
    unsigned int seed = blockIdx.x;
    for (unsigned int i = 1; i < static_cast<unsigned int>(cities); i++) {
      int j = static_cast<int>(LCG_random(&seed) * (cities - 1)) + 1;
      swap(px[i], px[j]);
      swap(py[i], py[j]);
    }
    px[cities] = px[0];
    py[cities] = py[0];
  }
  __syncthreads();

  int minchange;
  do {
    for (int i = threadIdx.x; i < cities; i += blockDim.x) buf[i] = -dist(i, i + 1);
    __syncthreads();

    minchange = 0;
    int mini = 1;
    int minj = 0;
    for (int ii = 0; ii < cities - 2; ii += blockDim.x) {
      int i = ii + threadIdx.x;
      float pxi0, pyi0, pxi1, pyi1, pxj1, pyj1;
      if (i < cities - 2) {
        minchange -= buf[i];
        pxi0 = px[i];
        pyi0 = py[i];
        pxi1 = px[i + 1];
        pyi1 = py[i + 1];
        pxj1 = px[cities];
        pyj1 = py[cities];
      }
      for (int jj = cities - 1; jj >= ii + 2; jj -= tilesize) {
        int bound = jj - tilesize + 1;
        for (int k = threadIdx.x; k < tilesize; k += blockDim.x) {
          if (k + bound >= ii + 2) {
            px_s[k] = px[k + bound];
            py_s[k] = py[k + bound];
            bf_s[k] = buf[k + bound];
          }
        }
        __syncthreads();

        int lower = bound;
        if (lower < i + 2) lower = i + 2;
        for (int j = jj; j >= lower; j--) {
          int jm = j - bound;
          float pxj0 = px_s[jm];
          float pyj0 = py_s[jm];
          int change = bf_s[jm]
            + int(sqrtf((pxi0 - pxj0) * (pxi0 - pxj0) + (pyi0 - pyj0) * (pyi0 - pyj0)))
            + int(sqrtf((pxi1 - pxj1) * (pxi1 - pxj1) + (pyi1 - pyj1) * (pyi1 - pyj1)));
          pxj1 = pxj0;
          pyj1 = pyj0;
          if (minchange > change) {
            minchange = change;
            mini = i;
            minj = j;
          }
        }
        __syncthreads();
      }

      if (i < cities - 2) {
        minchange += buf[i];
      }
    }
    __syncthreads();

    int change = buf_s[threadIdx.x] = minchange;
    if (threadIdx.x == 0) atomicAdd(climbs_d, 1);
    __syncthreads();

    int j = blockDim.x;
    do {
      int k = (j + 1) / 2;
      if ((threadIdx.x + k) < j) {
        int tmp = buf_s[threadIdx.x + k];
        if (change > tmp) change = tmp;
        buf_s[threadIdx.x] = change;
      }
      j = k;
      __syncthreads();
    } while (j > 1);

    if (minchange == buf_s[0]) {
      buf_s[1] = threadIdx.x;
    }
    __syncthreads();

    if (threadIdx.x == buf_s[1]) {
      buf_s[2] = mini + 1;
      buf_s[3] = minj;
    }
    __syncthreads();

    minchange = buf_s[0];
    mini = buf_s[2];
    int sum = buf_s[3] + mini;
    for (int i = threadIdx.x; (i + i) < sum; i += blockDim.x) {
      if (mini <= i) {
        int j = sum - i;
        swap(px[i], px[j]);
        swap(py[i], py[j]);
      }
    }
    __syncthreads();
  } while (minchange < 0);

  int term = 0;
  for (int i = threadIdx.x; i < cities; i += blockDim.x) {
    term += dist(i, i + 1);
  }
  buf_s[threadIdx.x] = term;
  __syncthreads();

  int j = blockDim.x;
  do {
    int k = (j + 1) / 2;
    if ((threadIdx.x + k) < j) {
      term += buf_s[threadIdx.x + k];
    }
    __syncthreads();
    if ((threadIdx.x + k) < j) {
      buf_s[threadIdx.x] = term;
    }
    j = k;
    __syncthreads();
  } while (j > 1);

  if (threadIdx.x == 0) {
    atomicMin(best_d, term);
  }
}
