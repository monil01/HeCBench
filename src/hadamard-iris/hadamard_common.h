#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include "utils.h"
#include "specials.h"
#include "reference.h"

#define GPU_CHECK(x) do { \
    int err = x; \
    if (err != hipSuccess) { \
        printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, "IRIS/HIP status failure"); \
        exit(1); \
    } \
} while (0)

struct HadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;

    index_t x_batch_stride;
    index_t out_batch_stride;

    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
};

void set_hadamard_params(HadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         void* x,
                         void* out,
                         float scale
                         )
{

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x;
    params.out_ptr = out;

    // All stride are in elements, not bytes.
    params.x_batch_stride = dim;  // seq_len * dim
    params.out_batch_stride = dim;  // seq_len * dim

    params.scale = scale;
}

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = 1 << kLogN;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    // We don't want to use more than 32 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 32 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_12(float x[kNChunks][12]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_12(x[c]); }
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_20(float x[kNChunks][20]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_20(x[c]); }
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_28(float x[kNChunks][28]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_28(x[c]); }
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_40(float x[kNChunks][40]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_40(x[c]); }
}
