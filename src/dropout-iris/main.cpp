#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

double checksum(const std::vector<float>& values) {
  double sum = 0.0;
  for (float value : values) sum += static_cast<double>(value);
  return sum;
}

long long mask_sum(const std::vector<uint8_t>& values) {
  long long sum = 0;
  for (uint8_t value : values) sum += static_cast<long long>(value);
  return sum;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const uint32_t nelem = static_cast<uint32_t>(std::strtoull(argv[1], nullptr, 10));
  const int repeat = std::atoi(argv[2]);
  if (nelem == 0 || repeat <= 0) {
    std::fprintf(stderr, "nelem and repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  constexpr size_t block_size = 256;
  constexpr size_t grid_vec1 = 512;
  constexpr size_t grid_vec2 = grid_vec1 / 2;
  constexpr size_t grid_vec4 = grid_vec1 / 4;
  constexpr uint64_t seed_first = 12345678ULL;
  constexpr uint64_t seed_second = 87654321ULL;

  std::vector<float> self_info(nelem, 0.1f);
  std::vector<float> ret_info(nelem, 0.0f);
  std::vector<uint8_t> mask_info(nelem, 0);

  iris::DMem self_mem(self_info.data(), self_info.size() * sizeof(float));
  iris::DMem ret_mem(ret_info.data(), ret_info.size() * sizeof(float));
  iris::DMem mask_mem(mask_info.data(), mask_info.size() * sizeof(uint8_t));

  auto run_kernel = [&](const char* name, size_t grid, int vec) -> double {
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 1; iter <= repeat; ++iter) {
      const float p = static_cast<float>(iter) / static_cast<float>(repeat);
      void* params[] = {
          &self_mem, &ret_mem, &mask_mem, const_cast<uint32_t*>(&nelem),
          const_cast<float*>(&p), const_cast<uint64_t*>(&seed_first),
          const_cast<uint64_t*>(&seed_second)};
      int pinfo[] = {
          iris_r, iris_w, iris_w, sizeof(nelem), sizeof(p),
          sizeof(seed_first), sizeof(seed_second)};
      size_t gws[] = {grid * block_size};
      size_t lws[] = {block_size};
      iris::Task task;
      task.kernel(name, 1, nullptr, gws, lws, 7, params, pinfo);
      if (vec == 4) {
        task.flush_out(ret_mem);
        task.flush_out(mask_mem);
      }
      task.submit(iris_gpu, nullptr, 1);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1.0e-9;
  };

  double t1 = run_kernel("fused_dropout_kernel_vec1", grid_vec1, 1);
  std::printf("Total kernel execution time (VEC1) %lf (s)\n", t1);
  double t2 = run_kernel("fused_dropout_kernel_vec2", grid_vec2, 2);
  std::printf("Total kernel execution time (VEC2) %lf (s)\n", t2);
  double t4 = run_kernel("fused_dropout_kernel_vec4", grid_vec4, 4);
  std::printf("Total kernel execution time (VEC4) %lf (s)\n", t4);

  std::printf("checksum %.9f\n", checksum(ret_info));
  std::printf("mask_sum %lld\n", mask_sum(mask_info));

  platform.finalize();
  return 0;
}
