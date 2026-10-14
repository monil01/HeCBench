#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "modP.h"

#define GPU_NUM_THREADS 64

static void intt_3_64k_modcrt_reference(uint32* dst, const uint64* src, int nttLen)
{
  const int blocks = nttLen / 512;
  for (int bidx = 0; bidx < blocks; bidx++) {
    uint64 buffer[512];
    for (int tidx = 0; tidx < GPU_NUM_THREADS; tidx++) {
      uint64 samples[8];
      const uint32 fmem = (bidx << 9) | ((tidx & 0x3E) << 3) | (tidx & 0x1);
      const uint32 tbuf = tidx << 3;
      for (int i = 0; i < 8; i++) {
        samples[i] = src[fmem | (i << 1)];
      }
      ntt8(samples);
      for (int i = 0; i < 8; i++) {
        buffer[tbuf | i] = _ls_modP(samples[i], ((tidx & 0x1) << 2) * i * 3);
      }
    }

    for (int tidx = 0; tidx < GPU_NUM_THREADS; tidx++) {
      uint64 samples[8];
      uint64 s8[8];
      const uint32 fbuf = ((tidx & 0x38) << 3) | (tidx & 0x7);
      const uint32 tmem = (bidx << 9) | ((tidx & 0x38) << 3) | (tidx & 0x7);
      for (int i = 0; i < 8; i++) {
        samples[i] = buffer[fbuf | (i << 3)];
      }
      for (int i = 0; i < 4; i++) {
        s8[2 * i] = _add_modP(samples[2 * i], samples[2 * i + 1]);
        s8[2 * i + 1] = _sub_modP(samples[2 * i], samples[2 * i + 1]);
      }
      for (int i = 0; i < 8; i++) {
        dst[(((tmem | (i << 3)) & 0xf) << 12) | ((tmem | (i << 3)) >> 4)] =
            static_cast<uint32>(_mul_modP(s8[i], 18446462594437939201UL, valP));
      }
    }
  }
}

static uint64_t checksum_u32(const uint32* values, int n)
{
  uint64_t checksum = 0;
  for (int i = 0; i < n; i++) {
    checksum += values[i];
  }
  return checksum;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int nttLen = 64 * 1024;
  const size_t ntt_size = static_cast<size_t>(nttLen) * sizeof(uint64);
  const size_t res_size = static_cast<size_t>(nttLen) * sizeof(uint32);

  uint64* ntt = static_cast<uint64*>(std::malloc(ntt_size));
  uint32* res = static_cast<uint32*>(std::malloc(res_size));
  uint32* ref = static_cast<uint32*>(std::malloc(res_size));
  if (!ntt || !res || !ref) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(ntt);
    std::free(res);
    std::free(ref);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (int i = 0; i < nttLen; i++) {
    uint64 hi = std::rand();
    uint64 lo = std::rand();
    ntt[i] = (hi << 32) | lo;
  }
  std::memset(res, 0, res_size);
  std::memset(ref, 0, res_size);

  intt_3_64k_modcrt_reference(ref, ntt, nttLen);
  const uint64_t ref_checksum = checksum_u32(ref, nttLen);

  iris::DMem mem_ntt(ntt, ntt_size);
  iris::DMem mem_res(res, res_size);

  const size_t gws[1] = { static_cast<size_t>(nttLen / 512) * GPU_NUM_THREADS };
  const size_t lws[1] = { GPU_NUM_THREADS };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = { &mem_res, &mem_ntt };
    int pinfo[2] = { iris_w, iris_r };
    task.kernel("intt_3_64k_modcrt", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 2, params, pinfo);
    task.flush_out(mem_res);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  const uint64_t checksum = checksum_u32(res, nttLen);
  std::printf("Checksum: %lu\n", checksum);
  const bool correct = checksum == ref_checksum && std::memcmp(res, ref, res_size) == 0;
  std::printf("%s\n", correct ? "PASS" : "FAIL");
  if (const char* path = std::getenv("HECBENCH_NTT_DUMP")) {
    FILE* output = std::fopen(path, "wb");
    if (!output || std::fwrite(res, sizeof(uint32), nttLen, output) != static_cast<size_t>(nttLen)) {
      std::fprintf(stderr, "NTT verification dump failed\n");
      if (output) std::fclose(output);
      platform.finalize();
      return 1;
    }
    std::fclose(output);
  }

  std::free(ntt);
  std::free(res);
  std::free(ref);
  platform.finalize();
  return correct ? 0 : 1;
}
