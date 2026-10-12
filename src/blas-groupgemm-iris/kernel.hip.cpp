#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void grouped_gemm_half(const __half* A,
                                              const __half* B,
                                              __half* C,
                                              const int* m,
                                              const int* n,
                                              const int* k,
                                              const size_t* a_offsets,
                                              const size_t* b_offsets,
                                              const size_t* c_offsets,
                                              const size_t* c_prefix,
                                              int groups,
                                              size_t total_c)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_c) return;

  int g = 0;
  while (g + 1 < groups && idx >= c_prefix[g + 1]) ++g;

  const size_t local = idx - c_prefix[g];
  const int M = m[g];
  const int N = n[g];
  const int K = k[g];
  const int row = local % M;
  const int col = local / M;
  if (col >= N) return;

  const size_t abase = a_offsets[g];
  const size_t bbase = b_offsets[g];
  float acc = 0.0f;
  for (int p = 0; p < K; ++p) {
    acc += __half2float(A[abase + row + (size_t)p * M]) *
           __half2float(B[bbase + p + (size_t)col * K]);
  }
  C[c_offsets[g] + local] = __float2half(acc);
}
