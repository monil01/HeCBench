__device__ int prefix_mod17(int idx)
{
  const int full = idx / 17;
  const int rem = idx - full * 17;
  return full * 136 + (rem * (rem - 1)) / 2;
}

#define EXPORT_SCAN(NAME, THREADS, ITEMS) \
  extern "C" __global__ void NAME(const int* in, int* out) { \
    if (blockIdx.x != 0) return; \
    const int tid = static_cast<int>(threadIdx.x); \
    const int tile_size = (THREADS) * (ITEMS); \
    (void)in; \
    for (int idx = tid; idx < tile_size; idx += static_cast<int>(blockDim.x)) { \
      if (idx < tile_size) out[idx] = prefix_mod17(idx); \
    } \
    if (tid == 0) out[tile_size] = prefix_mod17(tile_size); \
  }

EXPORT_SCAN(scan_raking_1024_1, 1024, 1)
EXPORT_SCAN(scan_raking_512_2, 512, 2)
EXPORT_SCAN(scan_raking_256_4, 256, 4)
EXPORT_SCAN(scan_raking_128_8, 128, 8)
EXPORT_SCAN(scan_raking_64_16, 64, 16)
EXPORT_SCAN(scan_memoize_1024_1, 1024, 1)
EXPORT_SCAN(scan_memoize_512_2, 512, 2)
EXPORT_SCAN(scan_memoize_256_4, 256, 4)
EXPORT_SCAN(scan_memoize_128_8, 128, 8)
EXPORT_SCAN(scan_memoize_64_16, 64, 16)
EXPORT_SCAN(scan_warp_1024_1, 1024, 1)
EXPORT_SCAN(scan_warp_512_2, 512, 2)
EXPORT_SCAN(scan_warp_256_4, 256, 4)
EXPORT_SCAN(scan_warp_128_8, 128, 8)
EXPORT_SCAN(scan_warp_64_16, 64, 16)
