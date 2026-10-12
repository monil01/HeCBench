#include <cuda_runtime.h>

#define F(x) (1.f + ((x) - 1.f) / 4.f)

constexpr int DIM = 30;
constexpr float OMEGA = 0.5f;
constexpr float c1 = 1.5f;
constexpr float c2 = 1.5f;
constexpr float phi = 3.1415f;

__device__ float fitness_function(float x[])
{
  float y1 = F(x[0]);
  float yn = F(x[DIM - 1]);
  float res = powf(sinf(phi * y1), 2.f) + powf(yn - 1, 2.f);
  for (int i = 0; i < DIM - 1; i++) {
    float y = F(x[i]);
    float yp = F(x[i + 1]);
    res += powf(y - 1.f, 2.f) * (1.f + 10.f * powf(sinf(phi * yp), 2.f));
  }
  return res;
}

extern "C" __global__ void kernelUpdateParticle(float* __restrict__ positions,
                                                 float* __restrict__ velocities,
                                                 const float* __restrict__ pBests,
                                                 const float* __restrict__ gBest,
                                                 int p,
                                                 float rp,
                                                 float rg)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p * DIM) return;
  velocities[i] = OMEGA * velocities[i] +
                  c1 * rp * (pBests[i] - positions[i]) +
                  c2 * rg * (gBest[i % DIM] - positions[i]);
  positions[i] += velocities[i];
}

extern "C" __global__ void kernelUpdatePBest(const float* __restrict__ positions,
                                             float* __restrict__ pBests,
                                             float* __restrict__ gBest,
                                             int p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p) return;
  i = i * DIM;
  float tempParticle1[DIM];
  float tempParticle2[DIM];
  for (int j = 0; j < DIM; j++) {
    tempParticle1[j] = positions[i + j];
    tempParticle2[j] = pBests[i + j];
  }
  if (fitness_function(tempParticle1) < fitness_function(tempParticle2)) {
    for (int j = 0; j < DIM; j++) pBests[i + j] = tempParticle1[j];
    if (fitness_function(tempParticle1) < 130.f) {
      for (int j = 0; j < DIM; j++) atomicAdd(gBest + j, tempParticle1[j]);
    }
  }
}
