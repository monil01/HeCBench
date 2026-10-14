#include <hip/hip_runtime.h>

#define VERTICES 600
#define BLOCK_SIZE_X 256

__device__ __forceinline__ int is_between(float a, float b, float c) {
  return (b > a) != (c > a);
}

template <int tile_size>
__device__ __forceinline__ void pnpoly_opt_body(int* __restrict__ bitmap,
                                                const float2* __restrict__ point,
                                                const float2* __restrict__ vertex,
                                                int n) {
  int i = blockIdx.x * BLOCK_SIZE_X * tile_size + threadIdx.x;
  if (i < n) {
    int c[tile_size];
    float2 lpoint[tile_size];
#pragma unroll
    for (int ti = 0; ti < tile_size; ti++) {
      c[ti] = 0;
      if (i + BLOCK_SIZE_X * ti < n) {
        lpoint[ti] = point[i + BLOCK_SIZE_X * ti];
      }
    }

    int k = VERTICES - 1;
    for (int j = 0; j < VERTICES; k = j++) {
      float2 vj = vertex[j];
      float2 vk = vertex[k];
      float slope = (vk.x - vj.x) / (vk.y - vj.y);
#pragma unroll
      for (int ti = 0; ti < tile_size; ti++) {
        float2 p = i + BLOCK_SIZE_X * ti < n ? lpoint[ti] : float2{0, 0};
        if (i + BLOCK_SIZE_X * ti < n && is_between(p.y, vj.y, vk.y) &&
            (p.x < slope * (p.y - vj.y) + vj.x)) {
          c[ti] = !c[ti];
        }
      }
    }

#pragma unroll
    for (int ti = 0; ti < tile_size; ti++) {
      if (i + BLOCK_SIZE_X * ti < n) bitmap[i + BLOCK_SIZE_X * ti] = c[ti];
    }
  }
}

extern "C" __global__ void pnpoly_base(int* __restrict__ bitmap,
                                        const float2* __restrict__ point,
                                        const float2* __restrict__ vertex,
                                        int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int c = 0;
    float2 p = point[i];
    int k = VERTICES - 1;
    for (int j = 0; j < VERTICES; k = j++) {
      float2 vj = vertex[j];
      float2 vk = vertex[k];
      float slope = (vk.x - vj.x) / (vk.y - vj.y);
      if (((vj.y > p.y) != (vk.y > p.y)) &&
          (p.x < slope * (p.y - vj.y) + vj.x)) {
        c = !c;
      }
    }
    bitmap[i] = c;
  }
}

extern "C" __global__ void pnpoly_opt_1(int* bitmap, const float2* point,
                                         const float2* vertex, int n) {
  pnpoly_opt_body<1>(bitmap, point, vertex, n);
}

extern "C" __global__ void pnpoly_opt_2(int* bitmap, const float2* point,
                                         const float2* vertex, int n) {
  pnpoly_opt_body<2>(bitmap, point, vertex, n);
}

extern "C" __global__ void pnpoly_opt_4(int* bitmap, const float2* point,
                                         const float2* vertex, int n) {
  pnpoly_opt_body<4>(bitmap, point, vertex, n);
}

extern "C" __global__ void pnpoly_opt_8(int* bitmap, const float2* point,
                                         const float2* vertex, int n) {
  pnpoly_opt_body<8>(bitmap, point, vertex, n);
}

extern "C" __global__ void pnpoly_opt_16(int* bitmap, const float2* point,
                                          const float2* vertex, int n) {
  pnpoly_opt_body<16>(bitmap, point, vertex, n);
}

extern "C" __global__ void pnpoly_opt_32(int* bitmap, const float2* point,
                                          const float2* vertex, int n) {
  pnpoly_opt_body<32>(bitmap, point, vertex, n);
}

extern "C" __global__ void pnpoly_opt_64(int* bitmap, const float2* point,
                                          const float2* vertex, int n) {
  pnpoly_opt_body<64>(bitmap, point, vertex, n);
}
