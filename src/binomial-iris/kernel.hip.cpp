#include <hip/hip_runtime.h>

#include <math.h>

#include "binomialOptions.h"
#include "realtype.h"

typedef struct
{
  real S;
  real X;
  real vDt;
  real puByDf;
  real pdByDf;
} __TOptionData;

__device__ inline float expiryCallValue(float S, float X, float vDt, int i)
{
  float d = S * __expf(vDt * (2.0f * i - NUM_STEPS)) - X;
  return (d > 0.0F) ? d : 0.0F;
}

#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)

extern "C" __global__ void binomialOptionsKernel(
    const __TOptionData *__restrict d_OptionData,
    real *__restrict d_CallValue)
{
  __shared__ real call_exchange[THREADBLOCK_SIZE + 1];

  const int tid = threadIdx.x;
  const real S = d_OptionData[blockIdx.x].S;
  const real X = d_OptionData[blockIdx.x].X;
  const real vDt = d_OptionData[blockIdx.x].vDt;
  const real puByDf = d_OptionData[blockIdx.x].puByDf;
  const real pdByDf = d_OptionData[blockIdx.x].pdByDf;

  real call[ELEMS_PER_THREAD + 1];
#pragma unroll
  for(int i = 0; i < ELEMS_PER_THREAD; ++i)
    call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

  if (tid == 0)
    call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

  int final_it = max(0, tid * ELEMS_PER_THREAD - 1);

#pragma unroll 16
  for(int i = NUM_STEPS; i > 0; --i)
  {
    call_exchange[tid] = call[0];
    __syncthreads();
    call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
    __syncthreads();

    if (i > final_it)
    {
#pragma unroll
      for(int j = 0; j < ELEMS_PER_THREAD; ++j)
        call[j] = puByDf * call[j + 1] + pdByDf * call[j];
    }
  }

  if (tid == 0)
    d_CallValue[blockIdx.x] = call[0];
}
