#include "hadamard_common.h"
template<typename Ktraits>
__device__
void fast_hadamard_transform_kernel(HadamardParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    constexpr int kNChunks = Ktraits::kNChunks;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");
    #if (defined(__GFX9__) || defined(__GFX8__))
    constexpr int kWarpSize = kNThreads < 64 ? kNThreads : 64;
    #else
    constexpr int kWarpSize = kNThreads < 32 ? kNThreads : 32;
    #endif
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");
    constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize, "kSmemExchangeSize should be a power of 2");
    static_assert(kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) == kNChunks * kNElts * sizeof(float));

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    // Shared memory.
    __shared__ __align__(16) char smem_[Ktraits::kSmemSize];
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int batch_id = blockIdx.x;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    float x_vals[kNChunks][kNElts];
    load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

    if constexpr (kNWarps > 1) {
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
        hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
    }

    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals_transposed[i][c] = x_vals[c][i]; }
        }
        if constexpr (kNChunks == 12) {
            hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 20) {
            hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 28) {
            hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 40) {
            hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
        } else {
            constexpr int kLogNChunks = cilog2(kNChunks);
            static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
            hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
        }
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = x_vals_transposed[i][c]; }
        }
    }

    store_output<kNChunks, kNElts, input_t>(out, x_vals, params.dim, params.scale);
}


extern "C" __global__ __launch_bounds__(1) void hadamard_f32_3(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<1,3,float>>(p); }

extern "C" __global__ __launch_bounds__(2) void hadamard_f32_4(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<2,4,float>>(p); }

extern "C" __global__ __launch_bounds__(4) void hadamard_f32_5(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<4,5,float>>(p); }

extern "C" __global__ __launch_bounds__(8) void hadamard_f32_6(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<8,6,float>>(p); }

extern "C" __global__ __launch_bounds__(16) void hadamard_f32_7(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16,7,float>>(p); }

extern "C" __global__ __launch_bounds__(32) void hadamard_f32_8(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<32,8,float>>(p); }

extern "C" __global__ __launch_bounds__(32) void hadamard_f32_9(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<32,9,float>>(p); }

extern "C" __global__ __launch_bounds__(128) void hadamard_f32_10(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<128,10,float>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f32_11(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,11,float>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f32_12(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,12,float>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f32_13(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,13,float>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f32_14(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,14,float>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f32_15(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,15,float>>(p); }

extern "C" __global__ __launch_bounds__(1) void hadamard_f16_3(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<1,3,__half>>(p); }

extern "C" __global__ __launch_bounds__(2) void hadamard_f16_4(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<2,4,__half>>(p); }

extern "C" __global__ __launch_bounds__(4) void hadamard_f16_5(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<4,5,__half>>(p); }

extern "C" __global__ __launch_bounds__(8) void hadamard_f16_6(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<8,6,__half>>(p); }

extern "C" __global__ __launch_bounds__(16) void hadamard_f16_7(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16,7,__half>>(p); }

extern "C" __global__ __launch_bounds__(32) void hadamard_f16_8(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<32,8,__half>>(p); }

extern "C" __global__ __launch_bounds__(32) void hadamard_f16_9(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<32,9,__half>>(p); }

extern "C" __global__ __launch_bounds__(128) void hadamard_f16_10(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<128,10,__half>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f16_11(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,11,__half>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f16_12(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,12,__half>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f16_13(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,13,__half>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f16_14(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,14,__half>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_f16_15(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,15,__half>>(p); }

extern "C" __global__ __launch_bounds__(1) void hadamard_bf_3(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<1,3,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(2) void hadamard_bf_4(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<2,4,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(4) void hadamard_bf_5(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<4,5,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(8) void hadamard_bf_6(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<8,6,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(16) void hadamard_bf_7(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16,7,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(32) void hadamard_bf_8(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<32,8,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(32) void hadamard_bf_9(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<32,9,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(128) void hadamard_bf_10(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<128,10,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_bf_11(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,11,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_bf_12(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,12,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_bf_13(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,13,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_bf_14(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,14,__hip_bfloat16>>(p); }

extern "C" __global__ __launch_bounds__(256) void hadamard_bf_15(void* x,void* out,int batch,int dim,int logn,int64_t xs,int64_t os,float scale) { HadamardParamsBase p; p.x_ptr=x; p.out_ptr=out; p.batch=batch; p.dim=dim; p.log_N=logn; p.x_batch_stride=xs; p.out_batch_stride=os; p.scale=scale; fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<256,15,__hip_bfloat16>>(p); }
