#include <cfloat>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() / (float)RAND_MAX * 2.f - 1.f; // range -1..1
    }
    return arr;
}

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

// warp-level reduction for summing values
__device__ inline float warpReduceSum(cooperative_groups::thread_block_tile<32> &warp, float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += warp.shfl_xor(val, offset);
    }
    return val;
}

// provide cache hints where possible
#define __stcs(ptr, val) patched_stcs(ptr, val)
#define __ldcs(ptr) patched_ldcs(ptr)
static __device__ __forceinline__ void patched_stcs(float *addr, float val) {
    __builtin_nontemporal_store(val, addr);
}

static __device__ __forceinline__ float patched_ldcs(const float *addr) {
    return __builtin_nontemporal_load(addr);
}

// ----------------------------------------------------------------------------
