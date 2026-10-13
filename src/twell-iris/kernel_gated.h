#include "torch_compat.h"
#include <hip/hip_runtime.h>
#define WARP_SIZE 32
#define STRIDE_8xWARP 256
namespace TWELL_GATED_T2D {
template <
    // full tile size
    const int T_n,
    // compressed tile size
    const int T_n_compressed,
    // number of tiles
    const int NUM_T_n,
    const int OUT_DIM
>
__device__ __forceinline__ void mm_t2d_kernel(
    // IN_DIM x OUT_DIM
    const __nv_bfloat16* IN_d,
    // IN_DIM x COMPRESSED_FEATURE_DIM
    const uint32_t* GATE_OUT_twell_packed_d,
    // FEATURE_DIM x OUT_DIM - transposed for coalesced access pattern
    const __nv_bfloat16* UP_transposed_d,
    // FEATURE_DIM x OUT_DIM
    const __nv_bfloat16* DOWN_d,
    // IN_DIM x OUT_DIM
    __nv_bfloat16* OUT_d
)
{
    static_assert((OUT_DIM % STRIDE_8xWARP) == 0, "OUT_DIM must be divisible by WARP_SIZE.");
    // each block computes one row
    constexpr int NUM_LOAD_ITERS = OUT_DIM / STRIDE_8xWARP;
    float OUT_accum[NUM_LOAD_ITERS][8] = {{0.0f}};
    __nv_bfloat162 IN_cached[NUM_LOAD_ITERS][4];

    IN_d += static_cast<size_t>(blockIdx.x) * OUT_DIM + threadIdx.x * 8;
    GATE_OUT_twell_packed_d += static_cast<size_t>(blockIdx.x) * T_n_compressed * NUM_T_n + threadIdx.x;
    // each iteration, each thread loads/stores 8 contiguous features
    // such that the whole warp processes STRIDE_8xWARP features
    UP_transposed_d += threadIdx.x * 8;
    DOWN_d += threadIdx.x * 8;
    OUT_d += static_cast<size_t>(blockIdx.x) * OUT_DIM + threadIdx.x * 8;

    // load cached input into registers
    #pragma unroll
    for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
    {
        *reinterpret_cast<uint4*>(&IN_cached[iter_idx][0]) =
            __ldcs(
                reinterpret_cast<const uint4*>(IN_d + iter_idx * STRIDE_8xWARP));
    }

    #pragma unroll 1
    for(int tile_idx = 0; tile_idx < NUM_T_n; ++tile_idx)
    {
        // each thread gets one scalar from the packed tile, giving the warp one coalesced access
        const int lane_tile_register = __ldcs(&GATE_OUT_twell_packed_d[tile_idx * T_n_compressed]);
        const int num_nonzeros = __shfl_sync(0xFFFFFFFFull, lane_tile_register, 0, 32);
        #pragma unroll 1
        for (int idx = 1; idx < num_nonzeros + 1; ++idx)
        {
            const uint32_t compressed_idx_bf16 = __shfl_sync(0xFFFFFFFFull, lane_tile_register, idx, 32);
            const uint32_t nonzero_idx = compressed_idx_bf16 & 0xFFFFu; // lower 16 bits for index

            float UP_OUT_accum = 0.0f;
            #pragma unroll
            // load 8 elements at a time
            for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
            {
                const uint4 packed_bfloats_x8 = *reinterpret_cast<const uint4*>(
                    UP_transposed_d + nonzero_idx * OUT_DIM + iter_idx * STRIDE_8xWARP);
                // unpack the 8 bfloat16s
                const __nv_bfloat162 packed_bfloats_1 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.x);
                __nv_bfloat162 scaled_bfloats_1 = __hmul2(IN_cached[iter_idx][0], packed_bfloats_1);
                float2 scaled_floats_1 = __bfloat1622float2(scaled_bfloats_1);
                // accumulate the dot product of the nonzero feature with these 8 features
                UP_OUT_accum += scaled_floats_1.x;
                UP_OUT_accum += scaled_floats_1.y;
                const __nv_bfloat162 packed_bfloats_2 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.y);
                __nv_bfloat162 scaled_bfloats_2 = __hmul2(IN_cached[iter_idx][1], packed_bfloats_2);
                float2 scaled_floats_2 = __bfloat1622float2(scaled_bfloats_2);
                UP_OUT_accum += scaled_floats_2.x;
                UP_OUT_accum += scaled_floats_2.y;

                const __nv_bfloat162 packed_bfloats_3 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.z);
                __nv_bfloat162 scaled_bfloats_3 = __hmul2(IN_cached[iter_idx][2], packed_bfloats_3);
                float2 scaled_floats_3 = __bfloat1622float2(scaled_bfloats_3);
                UP_OUT_accum += scaled_floats_3.x;
                UP_OUT_accum += scaled_floats_3.y;

                const __nv_bfloat162 packed_bfloats_4 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.w);
                __nv_bfloat162 scaled_bfloats_4 = __hmul2(IN_cached[iter_idx][3], packed_bfloats_4);
                float2 scaled_floats_4 = __bfloat1622float2(scaled_bfloats_4);
                UP_OUT_accum += scaled_floats_4.x;
                UP_OUT_accum += scaled_floats_4.y;
            }
            // butterfly shuffle - https://docs.nvidia.com/cuda/cuda-c-programming-guide/#reduction-across-a-warp
            #pragma unroll
            for (int butterfly_stride = WARP_SIZE/2; butterfly_stride > 0; butterfly_stride /= 2)
            {
                UP_OUT_accum += __shfl_xor_sync(0xFFFFFFFFull, UP_OUT_accum, butterfly_stride, 32);
            }

            const __nv_bfloat162 nonzero_feature = __bfloat162bfloat162(
                __hmul(
                    reinterpret_cast<const __nv_bfloat16*>(&compressed_idx_bf16)[1],
                    __float2bfloat16_rn(UP_OUT_accum)
                )
            ); // upper 16 bits for feature

            #pragma unroll
            // load 8 elements at a time
            for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
            {
                const uint4 packed_bfloats_x8 = *reinterpret_cast<const uint4*>(
                    DOWN_d + nonzero_idx * OUT_DIM + iter_idx * STRIDE_8xWARP);
                // unpack the 8 bfloat16s
                const __nv_bfloat162 packed_bfloats_1 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.x);
                __nv_bfloat162 scaled_bfloats_1 = __hmul2(nonzero_feature, packed_bfloats_1);
                float2 scaled_floats_1 = __bfloat1622float2(scaled_bfloats_1);
                // accumulate the dot product of the nonzero feature with these 8 features
                OUT_accum[iter_idx][0] += scaled_floats_1.x;
                OUT_accum[iter_idx][1] += scaled_floats_1.y;
                const __nv_bfloat162 packed_bfloats_2 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.y);
                __nv_bfloat162 scaled_bfloats_2 = __hmul2(nonzero_feature, packed_bfloats_2);
                float2 scaled_floats_2 = __bfloat1622float2(scaled_bfloats_2);
                OUT_accum[iter_idx][2] += scaled_floats_2.x;
                OUT_accum[iter_idx][3] += scaled_floats_2.y;

                const __nv_bfloat162 packed_bfloats_3 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.z);
                __nv_bfloat162 scaled_bfloats_3 = __hmul2(nonzero_feature, packed_bfloats_3);
                float2 scaled_floats_3 = __bfloat1622float2(scaled_bfloats_3);
                OUT_accum[iter_idx][4] += scaled_floats_3.x;
                OUT_accum[iter_idx][5] += scaled_floats_3.y;

                const __nv_bfloat162 packed_bfloats_4 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.w);
                __nv_bfloat162 scaled_bfloats_4 = __hmul2(nonzero_feature, packed_bfloats_4);
                float2 scaled_floats_4 = __bfloat1622float2(scaled_bfloats_4);
                OUT_accum[iter_idx][6] += scaled_floats_4.x;
                OUT_accum[iter_idx][7] += scaled_floats_4.y;
            }
        }
    }
    #pragma unroll
    for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
    {
        __nv_bfloat162  __align__(8) packed_bfloats_x8[4];
        packed_bfloats_x8[0] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][0], OUT_accum[iter_idx][1]);
        packed_bfloats_x8[1] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][2], OUT_accum[iter_idx][3]);
        packed_bfloats_x8[2] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][4], OUT_accum[iter_idx][5]);
        packed_bfloats_x8[3] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][6], OUT_accum[iter_idx][7]);

        __stcs(reinterpret_cast<uint4*>(OUT_d + iter_idx * STRIDE_8xWARP),
               *reinterpret_cast<uint4*>(packed_bfloats_x8));
    }
}
template <
    // full tile size
    const int T_n,
    // compressed tile size
    const int T_n_compressed,
    // number of tiles
    const int NUM_T_n,
    const int OUT_DIM
>
__device__ __forceinline__ void mm_t2d_high_precision_kernel(
    // IN_DIM x OUT_DIM
    const __nv_bfloat16* IN_d,
    // IN_DIM x COMPRESSED_FEATURE_DIM
    const uint32_t* GATE_OUT_twell_packed_d,
    // FEATURE_DIM x OUT_DIM - transposed for coalesced access pattern
    const __nv_bfloat16* UP_transposed_d,
    // FEATURE_DIM x OUT_DIM
    const __nv_bfloat16* DOWN_d,
    // IN_DIM x OUT_DIM
    __nv_bfloat16* OUT_d
)
{
    static_assert((OUT_DIM % STRIDE_8xWARP) == 0, "OUT_DIM must be divisible by WARP_SIZE.");
    static_assert(T_n_compressed == WARP_SIZE,
        "Warp sync twell 2 dense optimized for when the compressed tile is 32-dim.");
    // each block computes one row
    constexpr int NUM_LOAD_ITERS = OUT_DIM / STRIDE_8xWARP;
    float OUT_accum[NUM_LOAD_ITERS][8] = {{0.0f}};
    __nv_bfloat16 IN_cached[NUM_LOAD_ITERS][8];

    IN_d += static_cast<size_t>(blockIdx.x) * OUT_DIM + threadIdx.x * 8;
    GATE_OUT_twell_packed_d += static_cast<size_t>(blockIdx.x) * T_n_compressed * NUM_T_n + threadIdx.x;
    // each iteration, each thread loads/stores 8 contiguous features
    // such that the whole warp processes STRIDE_8xWARP features
    UP_transposed_d += threadIdx.x * 8;
    DOWN_d += threadIdx.x * 8;
    OUT_d += static_cast<size_t>(blockIdx.x) * OUT_DIM + threadIdx.x * 8;

    // load cached input into registers
    #pragma unroll
    for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
    {
        *reinterpret_cast<uint4*>(&IN_cached[iter_idx][0]) =
            __ldcs(
                reinterpret_cast<const uint4*>(IN_d + iter_idx * STRIDE_8xWARP));
    }

    #pragma unroll 1
    for(int tile_idx = 0; tile_idx < NUM_T_n; ++tile_idx)
    {
        // each thread gets one scalar from the packed tile, giving the warp one coalesced access
        const int lane_tile_register = __ldcs(&GATE_OUT_twell_packed_d[tile_idx * T_n_compressed]);
        const int num_nonzeros = __shfl_sync(0xFFFFFFFFull, lane_tile_register, 0, 32);
        #pragma unroll 1
        for (int idx = 1; idx < num_nonzeros + 1; ++idx)
        {
            const uint32_t compressed_idx_bf16 = __shfl_sync(0xFFFFFFFFull, lane_tile_register, idx, 32);
            const uint32_t nonzero_idx = compressed_idx_bf16 & 0xFFFFu; // lower 16 bits for index

            float UP_OUT_accum = 0.0f;
            #pragma unroll
            // load 8 elements at a time
            for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
            {
                const uint4 packed_bfloats_x8 = *reinterpret_cast<const uint4*>(
                    UP_transposed_d + nonzero_idx * OUT_DIM + iter_idx * STRIDE_8xWARP);

                #pragma unroll
                for(int elem_idx = 0; elem_idx < 8; ++elem_idx)
                {
                    float upcasted_weight = __bfloat162float(
                        reinterpret_cast<const __nv_bfloat16*>(
                            &packed_bfloats_x8)[elem_idx]
                        );
                    float upcasted_input = __bfloat162float(
                        IN_cached[iter_idx][elem_idx]);
                    // accumulate the dot product of the nonzero feature with these 8 features
                    UP_OUT_accum = __fmaf_rn(upcasted_weight, upcasted_input, UP_OUT_accum);
                }
            }
            // butterfly shuffle - https://docs.nvidia.com/cuda/cuda-c-programming-guide/#reduction-across-a-warp
            #pragma unroll
            for (int butterfly_stride = WARP_SIZE/2; butterfly_stride > 0; butterfly_stride /= 2)
            {
                UP_OUT_accum += __shfl_xor_sync(0xFFFFFFFFull, UP_OUT_accum, butterfly_stride, 32);
            }

            const float nonzero_feature = __bfloat162float(
                (reinterpret_cast<const __nv_bfloat16*>(
                    &compressed_idx_bf16)[1])
            ) * UP_OUT_accum;


            #pragma unroll
            // load 8 elements at a time
            for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
            {
                const uint4 packed_bfloats_x8 = *reinterpret_cast<const uint4*>(
                    DOWN_d + nonzero_idx * OUT_DIM + iter_idx * STRIDE_8xWARP);
                // unpack the 8 bfloat16s
                #pragma unroll
                for(int elem_idx = 0; elem_idx < 8; ++elem_idx)
                {
                    float upcasted_weight = __bfloat162float(
                        reinterpret_cast<const __nv_bfloat16*>(
                            &packed_bfloats_x8)[elem_idx]
                        );
                    // accumulate the dot product of the nonzero feature with these 8 features
                    OUT_accum[iter_idx][elem_idx] = __fmaf_rn(
                        upcasted_weight, nonzero_feature, OUT_accum[iter_idx][elem_idx]
                    );
                }
            }
        }
    }
    #pragma unroll
    for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
    {
        __nv_bfloat162  __align__(16) packed_bfloats_x8[4];
        packed_bfloats_x8[0] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][0], OUT_accum[iter_idx][1]);
        packed_bfloats_x8[1] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][2], OUT_accum[iter_idx][3]);
        packed_bfloats_x8[2] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][4], OUT_accum[iter_idx][5]);
        packed_bfloats_x8[3] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][6], OUT_accum[iter_idx][7]);

        __stcs(reinterpret_cast<uint4*>(OUT_d + iter_idx * STRIDE_8xWARP),
               *reinterpret_cast<uint4*>(packed_bfloats_x8));
    }
}
} // namespace

extern "C" __global__ __launch_bounds__(32) void t2d_bf16_22(
    const __nv_bfloat16* in,const uint32_t* packed,const __nv_bfloat16* up,
    const __nv_bfloat16* down,__nv_bfloat16* out) {
    TWELL_GATED_T2D::mm_t2d_kernel<256,32,22,2048>(in,packed,up,down,out);
}

extern "C" __global__ __launch_bounds__(32) void t2d_bf16_32(
    const __nv_bfloat16* in,const uint32_t* packed,const __nv_bfloat16* up,
    const __nv_bfloat16* down,__nv_bfloat16* out) {
    TWELL_GATED_T2D::mm_t2d_kernel<256,32,32,2048>(in,packed,up,down,out);
}

extern "C" __global__ __launch_bounds__(32) void t2d_fp32_22(
    const __nv_bfloat16* in,const uint32_t* packed,const __nv_bfloat16* up,
    const __nv_bfloat16* down,__nv_bfloat16* out) {
    TWELL_GATED_T2D::mm_t2d_high_precision_kernel<256,32,22,2048>(in,packed,up,down,out);
}

extern "C" __global__ __launch_bounds__(32) void t2d_fp32_32(
    const __nv_bfloat16* in,const uint32_t* packed,const __nv_bfloat16* up,
    const __nv_bfloat16* down,__nv_bfloat16* out) {
    TWELL_GATED_T2D::mm_t2d_high_precision_kernel<256,32,32,2048>(in,packed,up,down,out);
}

#undef WARP_SIZE
