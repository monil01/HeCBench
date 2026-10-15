#include <cuda_fp16.h>
#include <stdint.h>

constexpr int F = 9;

extern "C" __global__ void Code1x16MatVec32(
    const int4* __restrict__ A,
    const __half* __restrict__ B_all,
    __half* __restrict__ C_all,
    const int4* __restrict__ codebook,
    const int prob_m,
    const int prob_k,
    const int4 codebook_a_sizes,
    const int codebook_stride,
    const int vector_index)
{
  constexpr int WarpSize = 32;
  const int4* B = reinterpret_cast<const int4*>(B_all + 4096 * vector_index);
  __half* C = C_all + 12288 * vector_index;

  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / WarpSize) * blockIdx.x + (threadIdx.x / WarpSize);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    const int* codebook_size = &codebook_a_sizes.x;
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % WarpSize;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % WarpSize;

  __shared__ int4 sh_b[32 * F];
  float res = 0.0f;
  int iters = (prob_k / 8 + 8 * WarpSize - 1) / (8 * WarpSize);
  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < WarpSize * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) sh_b[F * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += WarpSize * 8;

    int b_sh_rd = F * (threadIdx.x % WarpSize);
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[4];
        int4 t = codebook[enc[i]];
        dec[0] = t.x;
        dec[1] = t.y;
        dec[2] = t.z;
        dec[3] = t.w;

        half2* a = reinterpret_cast<half2*>(&dec);
        half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
        half2 res2 = {};
#pragma unroll
        for (int j = 0; j < 4; j++) res2 = __hfma2(a[j], b[j], res2);
        res += __half2float(res2.x) + __half2float(res2.y);
        b_sh_rd++;
      }
      a_gl_rd += WarpSize;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = WarpSize / 2; i > 0; i /= 2) {
      res += __shfl_down_sync(0xffffffff, res, i);
    }
    if (threadIdx.x % WarpSize == 0) {
      C[c_gl_wr] = __float2half(res);
    }
  }
}
