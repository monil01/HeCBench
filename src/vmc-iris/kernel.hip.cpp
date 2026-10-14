#include <chrono>
#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

using namespace std;

using FLOAT = float;

#define CHECK(test) if (test != hipSuccess) throw "error";

const int NTHR_PER_BLK = 256;           // Number of CUDA threads per block
const int NBLOCK  = 56*4;               // Number of CUDA blocks (SMs on P100)
const int Npoint = NBLOCK*NTHR_PER_BLK; // No. of independent samples
const int Neq = 100000;                 // No. of generations to equilibrate
const int Ngen_per_block = 5000;        // No. of generations per block
const float DELTA = 2.0;                // Random step size

// Explicitly typed constants so can easily work with both floats and floats
static const FLOAT FOUR = 4.0;
static const FLOAT TWO  = 2.0;
static const FLOAT ONE  = 1.0;
static const FLOAT HALF = 0.5;
static const FLOAT ZERO = 0.0;

__device__ __forceinline__ float EXP(float x) {return expf(x);}
__device__ __forceinline__ double EXP(double x) {return exp(x);}
__device__ __forceinline__ float SQRT(float x) {return sqrtf(x);}
__device__ __forceinline__ double SQRT(double x) {return sqrt(x);}


__device__
float LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

__device__
void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}


extern "C" __global__ void SumWithinBlocks(const int n, const FLOAT* data, FLOAT* blocksums, int input_offset, int output_offset) {
  data+=input_offset; blocksums+=output_offset;
  int nthread = blockDim.x*gridDim.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ FLOAT sdata[512];  // max threads

  // Every thread in every block computes partial sum over rest of vector
  FLOAT st=ZERO;
  while (i < n) {
    st += data[i];
    i+=nthread;
  }
  sdata[threadIdx.x] = st;
  __syncthreads();

  // Now do binary tree sum within a block
  int tid = threadIdx.x;
  for (unsigned int s=128; s>0; s>>=1) {
    if (tid<s && (tid+s)<blockDim.x) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid==0) blocksums[blockIdx.x] = sdata[0];
}

__device__ __forceinline__ void compute_distances(FLOAT x1, FLOAT y1, FLOAT z1, FLOAT x2, FLOAT y2, FLOAT z2,
    FLOAT& r1, FLOAT& r2, FLOAT& r12) {
  r1 = SQRT(x1*x1 + y1*y1 + z1*z1);
  r2 = SQRT(x2*x2 + y2*y2 + z2*z2);
  FLOAT xx = x1-x2;
  FLOAT yy = y1-y2;
  FLOAT zz = z1-z2;
  r12 = SQRT(xx*xx + yy*yy + zz*zz);
}

__device__  __forceinline__ FLOAT wave_function(FLOAT x1, FLOAT y1, FLOAT z1, FLOAT x2, FLOAT y2, FLOAT z2) {
  FLOAT r1, r2, r12;
  compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);

  return (ONE + HALF*r12)*EXP(-TWO*(r1 + r2));
}

// Initialize random number generator
extern "C" __global__ void initran(unsigned int seed, unsigned int* states) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  states[i] = seed ^ i;
  LCG_random_init(&states[i]);
}

// ZERO stats counters on the GPU
extern "C" __global__ void zero_stats(int Npoint, FLOAT* stats) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  stats[0*Npoint+i] = ZERO; // r1
  stats[1*Npoint+i] = ZERO; // r2
  stats[2*Npoint+i] = ZERO; // r12
  stats[3*Npoint+i] = ZERO; // accept count
}

// initializes samples
extern "C" __global__ void initialize(FLOAT* __restrict__ x1,
                           FLOAT* __restrict__ y1,
                           FLOAT* __restrict__ z1,
                           FLOAT* __restrict__ x2,
                           FLOAT* __restrict__ y2,
                           FLOAT* __restrict__ z2,
                           FLOAT* __restrict__ psi,
                           unsigned int* __restrict__ states)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  x1[i] = (LCG_random(states+i) - HALF)*FOUR;
  y1[i] = (LCG_random(states+i) - HALF)*FOUR;
  z1[i] = (LCG_random(states+i) - HALF)*FOUR;
  x2[i] = (LCG_random(states+i) - HALF)*FOUR;
  y2[i] = (LCG_random(states+i) - HALF)*FOUR;
  z2[i] = (LCG_random(states+i) - HALF)*FOUR;
  psi[i] = wave_function(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]);
}

extern "C" __global__ void propagate(const int Npoint, const int nstep,
                          FLOAT* __restrict__  X1,
                          FLOAT* __restrict__  Y1,
                          FLOAT* __restrict__  Z1,
                          FLOAT* __restrict__  X2,
                          FLOAT* __restrict__  Y2,
                          FLOAT* __restrict__  Z2,
                          FLOAT* __restrict__  P,
                          FLOAT* __restrict__  stats,
                          unsigned int* __restrict__  states)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT x1 = X1[i];
  FLOAT y1 = Y1[i];
  FLOAT z1 = Z1[i];
  FLOAT x2 = X2[i];
  FLOAT y2 = Y2[i];
  FLOAT z2 = Z2[i];
  FLOAT p = P[i];

  for (int step=0; step<nstep; step++) {
    FLOAT x1new = x1 + (LCG_random(states+i)-HALF)*DELTA;
    FLOAT y1new = y1 + (LCG_random(states+i)-HALF)*DELTA;
    FLOAT z1new = z1 + (LCG_random(states+i)-HALF)*DELTA;
    FLOAT x2new = x2 + (LCG_random(states+i)-HALF)*DELTA;
    FLOAT y2new = y2 + (LCG_random(states+i)-HALF)*DELTA;
    FLOAT z2new = z2 + (LCG_random(states+i)-HALF)*DELTA;
    FLOAT pnew = wave_function(x1new, y1new, z1new, x2new, y2new, z2new);

    if (pnew*pnew > p*p*LCG_random(states+i)) {
      stats[3*Npoint+i]++; //naccept ++;
      p = pnew;
      x1 = x1new;
      y1 = y1new;
      z1 = z1new;
      x2 = x2new;
      y2 = y2new;
      z2 = z2new;
    }

    FLOAT r1, r2, r12;
    compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);

    stats[0*Npoint+i] += r1;
    stats[1*Npoint+i] += r2;
    stats[2*Npoint+i] += r12;
  }
  X1[i] = x1;
  Y1[i] = y1;
  Z1[i] = z1;
  X2[i] = x2;
  Y2[i] = y2;
  Z2[i] = z2;
  P[i] = p;
}
