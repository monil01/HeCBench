#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr uint64_t latencyMemAccessCnt = 1000000;
constexpr uint32_t _2MiB = 2 * 1024 * 1024;
constexpr uint32_t strideLen = 16;
constexpr uint32_t defaultSmCount = 120;

bool parse_u32(const char* text, uint32_t* value)
{
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0' || parsed == 0 || parsed > UINT32_MAX) {
    return false;
  }
  *value = static_cast<uint32_t>(parsed);
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  uint32_t smCount = defaultSmCount;
  if (argc > 2) {
    std::fprintf(stderr, "Usage: %s [target_block_count]\n", argv[0]);
    platform.finalize();
    return 1;
  }
  if (argc == 2 && !parse_u32(argv[1], &smCount)) {
    std::fprintf(stderr, "invalid target_block_count: %s\n", argv[1]);
    platform.finalize();
    return 1;
  }

  const uint64_t buffer_size = _2MiB;
  const uint64_t n_ptrs = buffer_size / sizeof(uintptr_t);
  std::vector<uintptr_t> buffer(n_ptrs, 0);
  uint32_t status = 0;

  iris::DMem mem_buffer(buffer.data(), buffer.size() * sizeof(uintptr_t));
  iris::DMem mem_status(&status, sizeof(status));

  {
    const size_t lws[1] = { 256 };
    const size_t gws[1] = { ((static_cast<size_t>(n_ptrs) + lws[0] - 1) / lws[0]) * lws[0] };
    void* params[3] = { &mem_buffer, const_cast<uint64_t*>(&n_ptrs), const_cast<uint32_t*>(&strideLen) };
    int pinfo[3] = { iris_w, sizeof(n_ptrs), sizeof(strideLen) };

    iris::Task task;
    task.kernel("init_pointer_chase", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  double latencySum = 0.0;
  const size_t lws[1] = { 1 };
  const size_t gws[1] = { smCount };
  const uint32_t access_count = static_cast<uint32_t>(latencyMemAccessCnt);
  uint32_t failures = 0;

  for (uint32_t targetBlock = 0; targetBlock < smCount; ++targetBlock) {
    status = 0;
    mem_status.update(&status);

    auto start = std::chrono::steady_clock::now();

    void* params[5] = {
      &mem_buffer,
      const_cast<uint64_t*>(&buffer_size),
      const_cast<uint32_t*>(&access_count),
      &targetBlock,
      &mem_status
    };
    int pinfo[5] = {
      iris_r,
      sizeof(buffer_size),
      sizeof(access_count),
      sizeof(targetBlock),
      iris_rw
    };

    iris::Task task;
    task.kernel("ptrChasingKernel", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 5, params, pinfo);
    task.flush_out(mem_status);
    task.submit(iris_gpu, nullptr, 1);

    auto end = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    latencySum += static_cast<double>(latency);

    mem_status.fetch(&status);
    failures += (status == 0);
  }

  const double lat = latencySum / (static_cast<double>(latencyMemAccessCnt) * smCount);
  std::printf("Latency per access on device: %lf (ns)\n", lat);
  if (failures != 0) {
    std::fprintf(stderr, "pointer chase status failures: %u\n", failures);
  }

  platform.finalize();
  return failures == 0 ? 0 : 1;
}
