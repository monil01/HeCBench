#include "kernel_d2t.h"
#include "kernel_gated.h"
#include "kernel_reference.h"
extern "C" __global__ void fill_bf16_kernel(__nv_bfloat16* p, size_t n, unsigned seed,
                                 float scale, float offset) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned h = static_cast<unsigned>(i) * 2654435761u ^ seed;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    float u01 = static_cast<float>(h & 0xFFFFu) / 65535.0f;
    p[i] = __float2bfloat16((u01 - 0.5f) * scale + offset);
}


extern "C" __global__ void copy_bf16(const __nv_bfloat16* in,__nv_bfloat16* out,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) out[i]=in[i];
}
extern "C" __global__ void wave_info(int* out) {if(threadIdx.x==0)*out=warpSize;}
