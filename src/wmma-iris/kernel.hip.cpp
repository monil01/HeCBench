/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
typedef half fp16;
typedef float fp32;
namespace wmma = rocwmma;
const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
const int TILE_M = 64, TILE_N = 64;
constexpr unsigned int WAVE_SIZE = 64;
// This kernel assumes that each thread block has warpSize threads
// D = alpha * (A x B) + beta * C
//
// In this example, we assume:
// : A is in row-major format     (M x K)
// : B is in col-major format     (K x N)
// : C, D are in row-major format (M x N)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
extern "C" __global__ void gemm_impl0(const uint32_t m, const uint32_t n, const uint32_t k,
                           fp16 const *__restrict__ a,
                           fp16 const *__restrict__ b,
                           fp32 const *c,
                           fp32 *d, const uint32_t lda, const uint32_t ldb,
                           const uint32_t ldc, const uint32_t ldd,
                           const fp32 alpha, const fp32 beta) {
  // Create frags
  auto fragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::row_major>();
  auto fragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::col_major>();
  auto fragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();
  auto fragAcc = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();

  wmma::fill_fragment(fragAcc, 0.0f);

  auto cRow = blockIdx.x * WMMA_M;
  auto cCol = blockIdx.y * WMMA_N;

  // Load the inputs
  for (int n = 0; n < k; n += WMMA_K) {
    // Because the mapping of elements to threads in a warp is opaque,
    // each thread just passes the address of the first element
    wmma::load_matrix_sync(fragA, a + cRow * lda + n, lda);
    wmma::load_matrix_sync(fragB, b + cCol * ldb + n, ldb);

    // Matrix multiply - accumulate using MFMA units
    wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);

    // Fetch C matrix
    wmma::load_matrix_sync(fragC, c + cRow * ldc + cCol, ldc, wmma::mem_row_major);

    // D = alpha * A x B + beta * C
    for (int i = 0; i < fragC.num_elements; ++i) {
      fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
    }
  }

  // Store to D (by a single wave)
  wmma::store_matrix_sync(d + cRow * ldd + cCol, fragC, ldd, wmma::mem_row_major);
}

// D = alpha * (A x B) + beta * C
//
// In this example, we assume:
// : A is in row-major format     (M x K)
// : B is in col-major format     (K x N)
// : C, D are in row-major format (M x N)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
//
extern "C" __global__ void gemm_impl1(const uint32_t m, const uint32_t n, const uint32_t k,
                           fp16 const *__restrict__ a,
                           fp16 const *__restrict__ b,
                           fp32 const *c,
                           fp32 *d, const uint32_t lda, const uint32_t ldb,
                           const uint32_t ldc, const uint32_t ldd,
                           const fp32 alpha, const fp32 beta) {
  // Create frags
  auto fragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::row_major>();
  auto fragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::col_major>();
  auto fragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();
  auto fragAcc = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();

  wmma::fill_fragment(fragAcc, 0.0f);

  // Map threadIdx to warpIdx
  auto warpIdx = threadIdx.x / WAVE_SIZE;
  auto warpIdy = threadIdx.y;

  // Target C block
  auto cRow = blockIdx.x * TILE_M + warpIdx * WMMA_M;
  auto cCol = blockIdx.y * TILE_N + warpIdy * WMMA_N;

  // Bounds check
  for (int n = 0; n < k; n += WMMA_K) {
    // Load the inputs
    wmma::load_matrix_sync(fragA, a + (cRow * lda + n), lda);
    wmma::load_matrix_sync(fragB, b + (cCol * ldb + n), ldb);

    // Matrix multiply - accumulate using MFMA units
    wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
  }

  // Fetch C matrix
  wmma::load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc,
                         wmma::mem_row_major);

  // D = alpha * A x B + beta * C
  for (int i = 0; i < fragC.num_elements; ++i) {
    fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
  }

  // Store to D
  wmma::store_matrix_sync(d + (cRow * ldd + cCol), fragC, ldd,
                          wmma::mem_row_major);
}


extern "C" __global__ void wave_info(int* out) { if(threadIdx.x == 0) *out = warpSize; }
