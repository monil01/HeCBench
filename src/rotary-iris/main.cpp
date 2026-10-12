#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int C10_WARP_SIZE = 32;
constexpr int num_threads() { return C10_WARP_SIZE * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

static bool run_rotary(int argc, char** argv, int repeat)
{
  constexpr int rotary_dim = 128;
  constexpr int rotary_pairs = rotary_dim / 2;
  const int num_tokens = block_work_size() * 10000;
  std::printf("Number of tokens: %d\n", num_tokens);
  std::printf("Rotary dimensions per token: %d\n", rotary_dim);

  const size_t numel = static_cast<size_t>(num_tokens) * rotary_pairs;
  const size_t bytes = numel * sizeof(float);
  std::vector<float> x1(numel);
  std::vector<float> x2(numel);
  std::vector<float> cosv(numel);
  std::vector<float> sinv(numel);
  std::vector<float> o1(numel, 0.0f);
  std::vector<float> o2(numel, 0.0f);

  std::mt19937 rng(12345);
  std::normal_distribution<float> input_dist(0.0f, 1.0f);
  for (size_t i = 0; i < numel; i++) {
    const float position = static_cast<float>(i / rotary_pairs);
    const float pair_index = static_cast<float>(i % rotary_pairs);
    const float angle = position / std::pow(10000.0f, 2.0f * pair_index / rotary_dim);
    x1[i] = input_dist(rng);
    x2[i] = input_dist(rng);
    cosv[i] = std::cos(angle);
    sinv[i] = std::sin(angle);
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  iris::DMem mem_x1(x1.data(), bytes);
  iris::DMem mem_x2(x2.data(), bytes);
  iris::DMem mem_cos(cosv.data(), bytes);
  iris::DMem mem_sin(sinv.data(), bytes);
  iris::DMem mem_o1(o1.data(), bytes);
  iris::DMem mem_o2(o2.data(), bytes);

  const size_t blocks = (numel + block_work_size() - 1) / block_work_size();
  std::printf("Number of thread blocks: %zu, thread block size: %d\n", blocks, num_threads());

  size_t gws[1] = {blocks * num_threads()};
  size_t lws[1] = {num_threads()};
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[] = {&mem_o1, &mem_o2, &mem_x1, &mem_x2, &mem_cos, &mem_sin, const_cast<size_t*>(&numel)};
    int pinfo[] = {iris_w, iris_w, iris_r, iris_r, iris_r, iris_r, sizeof(numel)};
    task.kernel("rotary_kernel", 1, nullptr, gws, lws, 7, params, pinfo);
    task.flush_out(mem_o1);
    task.flush_out(mem_o2);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

  mem_o1.fetch(o1.data());
  mem_o2.fetch(o2.data());
  platform.finalize();

  bool ok = true;
  for (size_t i = 0; i < numel; i++) {
    const float r1 = x1[i] * cosv[i] - x2[i] * sinv[i];
    const float r2 = x1[i] * sinv[i] + x2[i] * cosv[i];
    if ((r1 - o1[i]) > 1e-3f || (r2 - o2[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  return run_rotary(argc, argv, repeat) ? 0 : 2;
}
