#include <hip/hip_runtime.h>
#include <stdint.h>

#define FORCE_INLINE inline __attribute__((always_inline))
#define BIG_CONSTANT(x) (x##LU)

__device__ FORCE_INLINE uint64_t rotl64(uint64_t x, int8_t r)
{
  return (x << r) | (x >> (64 - r));
}

__device__ FORCE_INLINE uint64_t getblock64(const uint8_t* p, uint32_t i)
{
  uint64_t s = 0;
  for (uint32_t n = 0; n < 8; n++) {
    s |= (static_cast<uint64_t>(p[8 * i + n]) << (n * 8));
  }
  return s;
}

__device__ FORCE_INLINE uint64_t fmix64(uint64_t k)
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

__device__ void MurmurHash3_x64_128(const void* key,
                                    const uint32_t len,
                                    const uint32_t seed,
                                    void* out)
{
  const uint8_t* data = static_cast<const uint8_t*>(key);
  const uint32_t nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  for (uint32_t i = 0; i < nblocks; i++) {
    uint64_t k1 = getblock64(data, i * 2 + 0);
    uint64_t k2 = getblock64(data, i * 2 + 1);

    k1 *= c1;
    k1 = rotl64(k1, 31);
    k1 *= c2;
    h1 ^= k1;

    h1 = rotl64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = rotl64(k2, 33);
    k2 *= c1;
    h2 ^= k2;

    h2 = rotl64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;
  }

  const uint8_t* tail = data + nblocks * 16;

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (len & 15) {
    case 15: k2 ^= (static_cast<uint64_t>(tail[14])) << 48;
    case 14: k2 ^= (static_cast<uint64_t>(tail[13])) << 40;
    case 13: k2 ^= (static_cast<uint64_t>(tail[12])) << 32;
    case 12: k2 ^= (static_cast<uint64_t>(tail[11])) << 24;
    case 11: k2 ^= (static_cast<uint64_t>(tail[10])) << 16;
    case 10: k2 ^= (static_cast<uint64_t>(tail[9])) << 8;
    case 9:
      k2 ^= (static_cast<uint64_t>(tail[8])) << 0;
      k2 *= c2;
      k2 = rotl64(k2, 33);
      k2 *= c1;
      h2 ^= k2;

    case 8: k1 ^= (static_cast<uint64_t>(tail[7])) << 56;
    case 7: k1 ^= (static_cast<uint64_t>(tail[6])) << 48;
    case 6: k1 ^= (static_cast<uint64_t>(tail[5])) << 40;
    case 5: k1 ^= (static_cast<uint64_t>(tail[4])) << 32;
    case 4: k1 ^= (static_cast<uint64_t>(tail[3])) << 24;
    case 3: k1 ^= (static_cast<uint64_t>(tail[2])) << 16;
    case 2: k1 ^= (static_cast<uint64_t>(tail[1])) << 8;
    case 1:
      k1 ^= (static_cast<uint64_t>(tail[0])) << 0;
      k1 *= c1;
      k1 = rotl64(k1, 31);
      k1 *= c2;
      h1 ^= k1;
  }

  h1 ^= len;
  h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  static_cast<uint64_t*>(out)[0] = h1;
  static_cast<uint64_t*>(out)[1] = h2;
}

extern "C" __global__ void MurmurHash3_x64_128_kernel(const uint8_t* __restrict__ d_keys,
                                                       const uint32_t* __restrict__ d_length,
                                                       const uint32_t* __restrict__ length,
                                                       uint64_t* __restrict__ d_out,
                                                       const uint32_t numKeys)
{
  const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numKeys) {
    MurmurHash3_x64_128(d_keys + d_length[i], length[i], i, d_out + i * 2);
  }
}
