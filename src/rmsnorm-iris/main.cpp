#include <iris/iris.hpp>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> make_random_float(size_t n)
{
  std::vector<float> arr(n);
  for (size_t i = 0; i < n; i++) {
    arr[i] = std::rand() / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
  }
  return arr;
}

static void rmsnorm_forward_cpu(float* out, const float* inp, const float* gamma, int64_t n, int64_t h)
{
  constexpr float eps = 1e-5f;
  for (int64_t row = 0; row < n; row++) {
    const float* x = inp + row * h;
    float m = 0.0f;
    for (int64_t i = 0; i < h; i++) {
      m += x[i] * x[i];
    }
    m /= static_cast<float>(h);
    const float scale = 1.0f / std::sqrt(m + eps);
    float* out_row = out + row * h;
    for (int64_t i = 0; i < h; i++) {
      out_row[i] = x[i] * scale * gamma[i];
    }
  }
}

static bool validate_result(const std::vector<float>& gpu,
                            const std::vector<float>& ref,
                            const char* name,
                            float tolerance = 1e-5f)
{
  int nfaults = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    if (!std::isfinite(ref[i])) continue;
    const float t_eff = tolerance + std::fabs(ref[i]) * FLT_EPSILON;
    if (std::fabs(ref[i] - gpu[i]) > t_eff) {
      std::printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n",
                  name, i, ref[i], gpu[i]);
      nfaults++;
      if (nfaults >= 10) return false;
    }
  }
  return nfaults == 0;
}

static void launch_rmsnorm(iris::DMem& mem_inp,
                           iris::DMem& mem_gamma,
                           iris::DMem& mem_out,
                           int64_t inner_len,
                           float epsilon,
                           int block_size,
                           int64_t outer_len)
{
  size_t gws[1] = {static_cast<size_t>(outer_len) * static_cast<size_t>(block_size)};
  size_t lws[1] = {static_cast<size_t>(block_size)};
  void* params[] = {&mem_inp, &mem_gamma, &mem_out, &inner_len, &epsilon};
  int pinfo[] = {iris_r, iris_r, iris_w, sizeof(inner_len), sizeof(epsilon)};
  iris::Task task;
  task.kernel("rmsnorm_fwd_kernel", 1, nullptr, gws, lws, 5, params, pinfo);
  task.flush_out(mem_out);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t n = std::atoll(argv[1]);
  const int64_t h = std::atoll(argv[2]);
  const int repeat = std::atoi(argv[3]);
  std::srand(0);

  const size_t size = static_cast<size_t>(n) * static_cast<size_t>(h);
  std::vector<float> inp = make_random_float(size);
  std::vector<float> gamma = make_random_float(static_cast<size_t>(h));
  std::vector<float> ref(size);
  std::vector<float> out(size, 0.0f);
  rmsnorm_forward_cpu(ref.data(), inp.data(), gamma.data(), n, h);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  iris::DMem mem_inp(inp.data(), inp.size() * sizeof(float));
  iris::DMem mem_gamma(gamma.data(), gamma.size() * sizeof(float));
  iris::DMem mem_out(out.data(), out.size() * sizeof(float));

  const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  bool all_ok = true;
  for (int block_size : block_sizes) {
    std::printf("Checking block size %d.\n", block_size);
    launch_rmsnorm(mem_inp, mem_gamma, mem_out, h, 1e-5f, block_size, n);
    mem_out.fetch(out.data());
    all_ok = validate_result(out, ref, "out", 1e-5f) && all_ok;
  }

  if (!all_ok) {
    platform.finalize();
    return 2;
  }
  std::printf("All results match. Starting benchmarks.\n\n");

  for (int block_size : block_sizes) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
      launch_rmsnorm(mem_inp, mem_gamma, mem_out, h, 1e-5f, block_size, n);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;
    float elapsed_time = duration.count() / repeat;
    const size_t memory_ops = (2 * size + static_cast<size_t>(h)) * 4;
    const float memory_bandwidth = memory_ops / elapsed_time / 1e6f;
    std::printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n",
                block_size, elapsed_time, memory_bandwidth);
  }

  platform.finalize();
  return 0;
}
