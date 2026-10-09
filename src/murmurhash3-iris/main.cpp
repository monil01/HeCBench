#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define BLOCK_SIZE 256

#define FORCE_INLINE inline __attribute__((always_inline))
#define BIG_CONSTANT(x) (x##LU)

static FORCE_INLINE uint64_t rotl64(uint64_t x, int8_t r)
{
  return (x << r) | (x >> (64 - r));
}

static FORCE_INLINE uint64_t getblock64(const uint8_t* p, uint32_t i)
{
  uint64_t s = 0;
  for (uint32_t n = 0; n < 8; n++) {
    s |= (static_cast<uint64_t>(p[8 * i + n]) << (n * 8));
  }
  return s;
}

static FORCE_INLINE uint64_t fmix64(uint64_t k)
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

static void MurmurHash3_x64_128(const void* key,
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

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const uint32_t numKeys = static_cast<uint32_t>(std::atoi(argv[1]));
  const uint32_t repeat = static_cast<uint32_t>(std::atoi(argv[2]));

  std::srand(3);
  std::vector<uint32_t> length(numKeys);
  std::vector<std::vector<uint8_t>> keys(numKeys);
  std::vector<uint64_t> reference(static_cast<size_t>(numKeys) * 2);

  for (uint32_t i = 0; i < numKeys; i++) {
    length[i] = std::rand() % 10000;
    keys[i].resize(length[i]);
    for (uint32_t c = 0; c < length[i]; c++) {
      keys[i][c] = c % 256;
    }
    MurmurHash3_x64_128(keys[i].data(), length[i], i, &reference[2 * i]);
  }

  std::vector<uint32_t> offsets(static_cast<size_t>(numKeys) + 1);
  uint32_t total_length = 0;
  offsets[0] = 0;
  for (uint32_t i = 0; i < numKeys; i++) {
    total_length += length[i];
    offsets[i + 1] = total_length;
  }

  std::vector<uint8_t> packed_keys(total_length);
  for (uint32_t i = 0; i < numKeys; i++) {
    std::memcpy(packed_keys.data() + offsets[i], keys[i].data(), length[i]);
  }

  for (uint32_t i = 0; i < numKeys; i++) {
    assert(0 == std::memcmp(packed_keys.data() + offsets[i], keys[i].data(), length[i]));
  }

  std::vector<uint64_t> out(static_cast<size_t>(numKeys) * 2);

  iris::DMem mem_keys(packed_keys.data(), sizeof(uint8_t) * packed_keys.size());
  iris::DMem mem_offsets(offsets.data(), sizeof(uint32_t) * offsets.size());
  iris::DMem mem_length(length.data(), sizeof(uint32_t) * length.size());
  iris::DMem mem_out(out.data(), sizeof(uint64_t) * out.size());

  const size_t blocks = (static_cast<size_t>(numKeys) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (uint32_t n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[5] = { &mem_keys, &mem_offsets, &mem_length, &mem_out, const_cast<uint32_t*>(&numKeys) };
    int pinfo[5] = { iris_r, iris_r, iris_r, iris_w, sizeof(numKeys) };
    task.kernel("MurmurHash3_x64_128_kernel",
                1,
                nullptr,
                const_cast<size_t*>(gws),
                const_cast<size_t*>(lws),
                5,
                params,
                pinfo);
    if (n + 1 == repeat) {
      task.flush_out(mem_out);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  mem_out.fetch(out.data());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  bool error = false;
  for (uint32_t i = 0; i < numKeys; i++) {
    if (out[2 * i] != reference[2 * i] || out[2 * i + 1] != reference[2 * i + 1]) {
      error = true;
      break;
    }
  }
  std::printf("%s\n", error ? "FAIL" : "SUCCESS");

  platform.finalize();
  return 0;
}
