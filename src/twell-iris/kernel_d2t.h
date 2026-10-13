// AMD matrix-core port of the TwELL dense-to-twell (D2T) projection.
// Targets CDNA3 (gfx942, MI300) and CDNA4 (gfx950, MI350); both expose the
// bf16 16x16x16 MFMA instruction used here through the rocWMMA fragment API.
//
// The upstream CUDA D2T kernel (matmul_d2t.cu) is a Hopper-only design built on
// warpgroup WGMMA, TMA tensor maps, thread-block clusters and mbarriers. None
// of those primitives exist on AMD CDNA, so this file is a performance-oriented
// reimplementation of the *same* mathematical operation using AMD matrix cores
// (MFMA, via the rocWMMA fragment API) instead of a line-by-line PTX port.
//
// Operation (per layer):
//     C = relu(A @ B^T)                        (dense gate pre-activation)
// where A is (M x K) bf16 activations (row-major), B is (N x K) bf16 gate
// weights (row-major, so B^T is the K x N operand), and C is the (M x N) gate
// pre-activation. Rather than materializing the dense C, the kernel packs the
// positive entries of every 256-wide N-tile of each row into the TwELL packed
// format consumed by matmul_t2d.cu / matmul_gated_t2d.cu:
//
//   For row m and N-tile t (t in [0, N/256)), a run of T_n_compressed = 32
//   uint32 slots is written at C_packed[m * (N/256) * 32 + t * 32 + slot]:
//     slot 0        : number of positive activations in the tile (nnz)
//     slot 1..nnz   : (global_n_index & 0xFFFF) | (bf16(value) << 16)
//   Slot writes wrap modulo 31 (matching the TS8 LOOP_OVERFLOW_STORAGE path);
//   the benchmark keeps gate sparsity within the 31-slot budget.
//
// Tiling: one block computes a 32(row) x 256(col) output tile. 256 threads =
// four 64-lane wavefronts; each wavefront owns a 32x64 sub-tile (two 16x16 row
// fragments x four 16x16 col fragments) and streams the K dimension through the
// bf16 16x16x16 MFMA. Accumulated fp32 results are staged to LDS, then one lane
// per row scans its 256 columns and emits the packed representation.

#include "torch_compat.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>

#include <cstdint>
#include <cstdio>
#include <unordered_map>

namespace TWELL_D2T {

// tile geometry
static constexpr int T_m = 32;    // rows per block
static constexpr int T_n = 256;   // cols per block (one packed N-tile)
static constexpr int WMMA = 16;   // MFMA fragment edge (16x16x16 bf16)
static constexpr int BK = 16;     // K step per MFMA
static constexpr int WAVE_SIZE = 64;
static constexpr int NUM_WAVES = 4;
static constexpr int BLOCK_THREADS = WAVE_SIZE * NUM_WAVES;  // 256
static constexpr int T_n_compressed = T_n / 8;               // 32
static constexpr int ROW_TILES = T_m / WMMA;                 // 2
static constexpr int COL_TILES_PER_WAVE = (T_n / WMMA) / NUM_WAVES;  // 4

using rocwmma::bfloat16_t;

// computes one 32x256 relu'd A@B^T tile with MFMA and packs positive entries.
// inputs: A (M x K, row-major bf16), B (N x K, row-major bf16), logical M/K/N.
// output: TwELL packed positive activations written to C_packed.
extern "C" __global__ __launch_bounds__(BLOCK_THREADS) void mm_mfma_d2t_kernel(
    const bfloat16_t* __restrict__ A,
    const bfloat16_t* __restrict__ B,
    uint32_t* __restrict__ C_packed,
    const size_t M, const int K, const int N)
{
    const int wave = threadIdx.x / WAVE_SIZE;   // 0..3
    const int m0 = blockIdx.y * T_m;            // first row of the block tile
    const int n0 = blockIdx.x * T_n;            // first col of the block tile
    const int wave_n0 = n0 + wave * (T_n / NUM_WAVES);  // this wave's 64 cols

    using AFrag = rocwmma::fragment<rocwmma::matrix_a, WMMA, WMMA, WMMA,
                                    bfloat16_t, rocwmma::row_major>;
    using BFrag = rocwmma::fragment<rocwmma::matrix_b, WMMA, WMMA, WMMA,
                                    bfloat16_t, rocwmma::col_major>;
    using CFrag = rocwmma::fragment<rocwmma::accumulator, WMMA, WMMA, WMMA, float>;

    CFrag acc[ROW_TILES][COL_TILES_PER_WAVE];
#pragma unroll
    for (int r = 0; r < ROW_TILES; ++r)
#pragma unroll
        for (int c = 0; c < COL_TILES_PER_WAVE; ++c)
            rocwmma::fill_fragment(acc[r][c], 0.0f);

    // stream the K dimension through the matrix core
    for (int k = 0; k < K; k += BK) {
        AFrag a[ROW_TILES];
#pragma unroll
        for (int r = 0; r < ROW_TILES; ++r) {
            // A row-major: element[i][kk] at (m0 + r*16 + i)*K + (k + kk)
            rocwmma::load_matrix_sync(a[r], A + (size_t)(m0 + r * WMMA) * K + k, K);
        }
#pragma unroll
        for (int c = 0; c < COL_TILES_PER_WAVE; ++c) {
            BFrag b;
            // B (N x K) viewed col-major with ld=K yields B^T: element[kk][j]
            // = B[(wave_n0 + c*16 + j)*K + (k + kk)].
            rocwmma::load_matrix_sync(
                b, B + (size_t)(wave_n0 + c * WMMA) * K + k, K);
#pragma unroll
            for (int r = 0; r < ROW_TILES; ++r) {
                rocwmma::mma_sync(acc[r][c], a[r], b, acc[r][c]);
            }
        }
    }

    // stage the fp32 tile to LDS in row-major [32][256]
    __shared__ float tile[T_m][T_n];
#pragma unroll
    for (int r = 0; r < ROW_TILES; ++r)
#pragma unroll
        for (int c = 0; c < COL_TILES_PER_WAVE; ++c) {
            float* dst = &tile[r * WMMA][wave * (T_n / NUM_WAVES) + c * WMMA];
            rocwmma::store_matrix_sync(dst, acc[r][c], T_n, rocwmma::mem_row_major);
        }

    __syncthreads();

    // one lane per row packs the positive entries of its 256-wide tile
    const int row = threadIdx.x;
    if (row < T_m) {
        const int tile_n = blockIdx.x;                // packed N-tile index
        const int num_tiles_n = N / T_n;
        uint32_t* out = C_packed +
            (size_t)(m0 + row) * num_tiles_n * T_n_compressed +
            (size_t)tile_n * T_n_compressed;
        uint32_t count = 0;
#pragma unroll 4
        for (int col = 0; col < T_n; ++col) {
            const float v = tile[row][col];
            if (v > 0.0f) {
                const unsigned short bits =
                    __bfloat16_as_ushort(__float2bfloat16(v));
                const uint32_t idx = (uint32_t)(tile_n * T_n + col) & 0xFFFFu;
                // wrap within this tile's payload slots [1, 31] so an over-budget
                // (lossy) tile never corrupts a neighboring row's count slot.
                const uint32_t slot = (count % (T_n_compressed - 1)) + 1;
                out[slot] = idx | ((uint32_t)bits << 16);
                ++count;
            }
        }
        out[0] = count;
    }
}


} // namespace TWELL_D2T
