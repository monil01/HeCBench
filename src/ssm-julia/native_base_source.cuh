// Modified codes from Claude Free

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "../ssm-cuda/reference.h"

#define GPU_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static constexpr int kBatch  = 32;
static constexpr int kDim    = 2048; // D - feature / model dimension
static constexpr int kDstate =  8;   // N - SSM state size
static constexpr int kSeqLen = 1024; // L - sequence length (must be divisible
                                     //     by kNThreads*kNItems for even-len path)

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.f) ? x : __logf(1.f + __expf(x));
}
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.f + __expf(-x));
}

__global__ void selective_scan_base(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D_vec,
    const float* __restrict__ delta_bias,
    const float* __restrict__ z,         // may be nullptr
    bool   delta_softplus,
    int    batch, int dim, int dstate, int seqlen,
    float* __restrict__ ssm_states,
    float* __restrict__ y
) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch || d >= dim) return;

    float db = delta_bias[d];
    float Dv = D_vec[d];
    float h[kDstate];
    for (int n = 0; n < dstate; ++n)
       h[n] = ssm_states[b*dim*dstate + d*dstate + n];
    
    for (int l = 0; l < seqlen; ++l) {
        float u_v  = u    [b*dim*seqlen + d*seqlen + l];
        float dt   = delta[b*dim*seqlen + d*seqlen + l] + db;
        if (delta_softplus) dt = softplus_f(dt);

        float yv = Dv * u_v;
        for (int n = 0; n < dstate; ++n) {
            float dA = __expf(dt * A[d*dstate + n]);
            float dB = dt * B[b*dstate*seqlen + n*seqlen + l];
            h[n] = dA * h[n] + dB * u_v;
            yv  += C[b*dstate*seqlen + n*seqlen + l] * h[n];
        }
        if (z) yv *= silu_f(z[b*dim*seqlen + d*seqlen + l]);
        y[b*dim*seqlen + d*seqlen + l] = yv;
    }
    for (int n = 0; n < dstate; ++n)
      ssm_states[b*dim*dstate + d*dstate + n] = h[n];
}

