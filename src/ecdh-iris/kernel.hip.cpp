#include <hip/hip_runtime.h>

#define ESS __device__
#include "../ecdh-cuda/ecdh.h"
#undef ESS

extern "C" __global__ void k_slow_repeat(
    const int sk,
    const int P_x,
    const int P_y,
    int* __restrict__ T_x,
    int* __restrict__ T_y,
    const unsigned int m,
    const int a,
    const int num_pk,
    const int repeat)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pk) {
    for (int r = 0; r < repeat; r++) {
      make_pk_slow(sk, P_x, P_y, T_x + i, T_y + i, m, a);
    }
  }
}

extern "C" __global__ void k_fast_repeat(
    const int sk,
    const int P_x,
    const int P_y,
    int* __restrict__ T_x,
    int* __restrict__ T_y,
    const unsigned int m,
    const int a,
    const int num_pk,
    const int repeat)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pk) {
    for (int r = 0; r < repeat; r++) {
      make_pk_fast(sk, P_x, P_y, T_x + i, T_y + i, m, a);
    }
  }
}
