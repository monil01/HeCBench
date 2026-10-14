#include <hip/hip_runtime.h>

#define bidx blockIdx.x
#define tidx threadIdx.x

#include "modP.h"

extern "C" __global__ void intt_3_64k_modcrt(uint32* __restrict__ dst,
                                             const uint64* __restrict__ src)
{
  __shared__ uint64 buffer[512];
  uint64 samples[8], s8[8];
  uint32 fmem, tmem, fbuf, tbuf;
  fmem = (bidx << 9) | ((tidx & 0x3E) << 3) | (tidx & 0x1);
  tbuf = tidx << 3;
  fbuf = ((tidx & 0x38) << 3) | (tidx & 0x7);
  tmem = (bidx << 9) | ((tidx & 0x38) << 3) | (tidx & 0x7);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    samples[i] = src[fmem | (i << 1)];
  }
  ntt8(samples);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    buffer[tbuf | i] = _ls_modP(samples[i], ((tidx & 0x1) << 2) * i * 3);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 8; i++) {
    samples[i] = buffer[fbuf | (i << 3)];
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    s8[2 * i] = _add_modP(samples[2 * i], samples[2 * i + 1]);
    s8[2 * i + 1] = _sub_modP(samples[2 * i], samples[2 * i + 1]);
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    dst[(((tmem | (i << 3)) & 0xf) << 12) | ((tmem | (i << 3)) >> 4)] =
        (uint32)(_mul_modP(s8[i], 18446462594437939201UL, valP));
  }
}
