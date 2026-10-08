#include <cuda_runtime.h>

#include <stdint.h>

#include "chacha20.h"

__device__ void hex_to_raw(const char* src, const int n, uint8_t* dst,
                           const uint8_t* char_to_uint)
{
  for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
    uint8_t hi = char_to_uint[static_cast<unsigned char>(src[i * 2 + 0])];
    uint8_t lo = char_to_uint[static_cast<unsigned char>(src[i * 2 + 1])];
    dst[i] = (hi << 4) | lo;
  }
}

extern "C" __global__ void test_keystreams(
    const char* __restrict__ text_key,
    const char* __restrict__ text_nonce,
    const char* __restrict__ text_keystream,
    const uint8_t* __restrict__ char_to_uint,
    uint8_t* __restrict__ raw_key,
    uint8_t* __restrict__ raw_nonce,
    uint8_t* __restrict__ raw_keystream,
    uint8_t* __restrict__ result,
    const int text_key_size,
    const int text_nonce_size,
    const int text_keystream_size)
{
  hex_to_raw(text_key, text_key_size, raw_key, char_to_uint);
  hex_to_raw(text_nonce, text_nonce_size, raw_nonce, char_to_uint);
  hex_to_raw(text_keystream, text_keystream_size, raw_keystream, char_to_uint);

  if (threadIdx.x == 0) {
    Chacha20 chacha(raw_key, raw_nonce);
    chacha.crypt(result, text_keystream_size / 2);
  }
}
