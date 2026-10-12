#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

static uint32_t xorshift32(uint32_t state)
{
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static float uint32_to_uniform(uint32_t x)
{
  return (x >> 8) * 0x1.0p-24f;
}

static uint16_t float_to_bf16(float v)
{
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7fffU + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

static float bf16_to_float(uint16_t v)
{
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

template <typename T, int Kind>
static float to_float32(T v)
{
  if constexpr (Kind == 0) {
    return v;
  } else if constexpr (Kind == 1) {
    return static_cast<float>(v);
  } else {
    return bf16_to_float(v);
  }
}

template <typename T, int Kind>
static T from_float32(float v)
{
  if constexpr (Kind == 0) {
    return v;
  } else if constexpr (Kind == 1) {
    return static_cast<_Float16>(v);
  } else {
    return float_to_bf16(v);
  }
}

template <typename T, int Kind>
static void uniform_fill(std::vector<T>& data, float low, float high, uint32_t seed)
{
  for (uint64_t idx = 0; idx < data.size(); idx++) {
    uint32_t r = xorshift32(seed ^ static_cast<uint32_t>(idx));
    float u = uint32_to_uniform(r);
    float v = low + (high - low) * u;
    data[idx] = from_float32<T, Kind>(v);
  }
}

static void uniform_fill_float(std::vector<float>& data, float low, float high,
                               uint32_t seed)
{
  for (uint64_t idx = 0; idx < data.size(); idx++) {
    uint32_t r = xorshift32(seed ^ static_cast<uint32_t>(idx));
    float u = uint32_to_uniform(r);
    data[idx] = low + (high - low) * u;
  }
}

template <typename T, int Kind>
static void reference(std::vector<T>& output,
                      const std::vector<T>& prefix_output,
                      const std::vector<T>& suffix_output,
                      std::vector<float>& lse,
                      const std::vector<float>& prefix_lse,
                      const std::vector<float>& suffix_lse,
                      uint64_t num_tokens,
                      uint32_t num_heads,
                      uint32_t head_size)
{
  for (uint64_t t = 0; t < num_tokens; ++t) {
    for (uint32_t h = 0; h < num_heads; ++h) {
      const uint64_t lse_idx = t * num_heads + h;
      const float inf = std::numeric_limits<float>::infinity();
      float p_lse = prefix_lse[lse_idx];
      float s_lse = suffix_lse[lse_idx];
      if (p_lse == inf) p_lse = -inf;
      if (s_lse == inf) s_lse = -inf;
      const float max_lse = std::max(p_lse, s_lse);
      const float p_exp = std::exp(p_lse - max_lse);
      const float s_exp = std::exp(s_lse - max_lse);
      const float out_se = p_exp + s_exp;
      lse[lse_idx] = std::log(out_se) + max_lse;
      const float p_scale = p_exp / out_se;
      const float s_scale = s_exp / out_se;
      const uint64_t base = t * num_heads * head_size + h * head_size;
      for (uint32_t d = 0; d < head_size; ++d) {
        float sum = to_float32<T, Kind>(prefix_output[base + d]) * p_scale +
                    to_float32<T, Kind>(suffix_output[base + d]) * s_scale;
        output[base + d] = from_float32<T, Kind>(sum);
      }
    }
  }
}

static void run_kernel(const char* name, size_t gws, size_t lws, int nparams,
                       void** params, int* pinfo, iris::DMem& output,
                       iris::DMem& lse)
{
  iris::Task task;
  size_t global[1] = {gws};
  size_t local[1] = {lws};
  task.kernel(name, 1, nullptr, global, local, nparams, params, pinfo);
  task.flush_out(output);
  task.flush_out(lse);
  task.submit(iris_gpu, nullptr, 1);
}

template <typename T, int Kind>
static bool run_case(const char* label, const char* kernel_name, int32_t repeat,
                     uint64_t num_tokens, uint32_t num_heads,
                     uint32_t head_size)
{
  const uint64_t output_size = num_tokens * num_heads * head_size;
  const uint64_t lse_size = num_tokens * num_heads;
  const size_t output_size_bytes = output_size * sizeof(T);
  const size_t lse_size_bytes = lse_size * sizeof(float);

  std::vector<T> h_prefix_output(output_size), h_suffix_output(output_size);
  std::vector<T> h_output(output_size), r_output(output_size);
  std::vector<float> h_prefix_lse(lse_size), h_suffix_lse(lse_size);
  std::vector<float> h_lse(lse_size), r_lse(lse_size);

  const int fill_threads = 256;
  const float scale = 1.f / std::sqrt(static_cast<float>(head_size));
  const uint32_t seed = 1234;
  (void)fill_threads;
  uniform_fill<T, Kind>(h_prefix_output, -scale, scale, seed);
  uniform_fill<T, Kind>(h_suffix_output, -scale, scale, seed);
  uniform_fill_float(h_prefix_lse, -scale, scale, seed);
  uniform_fill_float(h_suffix_lse, -scale, scale, seed);

  reference<T, Kind>(r_output, h_prefix_output, h_suffix_output, r_lse,
                     h_prefix_lse, h_suffix_lse, num_tokens, num_heads,
                     head_size);

  const uint32_t pack_size = 16 / sizeof(T);
  if (head_size % pack_size != 0) {
    std::printf("headsize must be multiple of pack_size %d\n", pack_size);
    return false;
  }
  const uint32_t threads_per_head = head_size / pack_size;
  const uint64_t total_threads = num_tokens * num_heads * threads_per_head;
  constexpr uint32_t NUM_THREADS = 128;
  const size_t blocks = (total_threads + NUM_THREADS - 1) / NUM_THREADS;

  iris::DMem d_output(h_output.data(), output_size_bytes);
  iris::DMem d_prefix_output(h_prefix_output.data(), output_size_bytes);
  iris::DMem d_suffix_output(h_suffix_output.data(), output_size_bytes);
  iris::DMem d_lse(h_lse.data(), lse_size_bytes);
  iris::DMem d_prefix_lse(h_prefix_lse.data(), lse_size_bytes);
  iris::DMem d_suffix_lse(h_suffix_lse.data(), lse_size_bytes);

  void* params[9] = {&d_output, &d_prefix_output, &d_suffix_output, &d_lse,
                     &d_prefix_lse, &d_suffix_lse,
                     &num_tokens, &num_heads, &head_size};
  int pinfo[9] = {iris_w, iris_r, iris_r, iris_w, iris_r, iris_r,
                  sizeof(num_tokens), sizeof(num_heads), sizeof(head_size)};

  for (int i = 0; i < 100; i++) {
    run_kernel(kernel_name, blocks * NUM_THREADS, NUM_THREADS, 9, params,
               pinfo, d_output, d_lse);
  }
  d_output.fetch(h_output.data());
  d_lse.fetch(h_lse.data());

  bool ok = true;
  for (uint64_t i = 0; i < output_size; i++) {
    if (std::fabs(to_float32<T, Kind>(h_output[i]) -
                  to_float32<T, Kind>(r_output[i])) > 1e-3f) {
      ok = false;
      break;
    }
  }
  for (uint64_t i = 0; i < lse_size; i++) {
    if (std::fabs(h_lse[i] - r_lse[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  std::printf("output dtype %s: %s\n", label, ok ? "PASS" : "FAIL");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_kernel(kernel_name, blocks * NUM_THREADS, NUM_THREADS, 9, params,
               pinfo, d_output, d_lse);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the kernel: %f (us)\n",
              (time * 1e-3f) / repeat);
  return ok;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <number of tokens> <number of heads> <head size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  uint64_t num_tokens = std::strtoull(argv[1], nullptr, 10);
  uint32_t num_heads = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
  uint32_t head_size = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
  int32_t repeat = std::atoi(argv[4]);

  std::printf("\n#tokens %zu, #heads %d, head dimension %d\n",
              static_cast<size_t>(num_tokens), num_heads, head_size);
  bool ok = true;
  ok &= run_case<float, 0>("FP32", "merge_attn_states_float", repeat,
                           num_tokens, num_heads, head_size);
  ok &= run_case<_Float16, 1>("FP16", "merge_attn_states_half", repeat,
                              num_tokens, num_heads, head_size);
  ok &= run_case<uint16_t, 2>("BF16", "merge_attn_states_bfloat16", repeat,
                              num_tokens, num_heads, head_size);
  std::printf("----------------------------------------------------\n");
  platform.finalize();
  return ok ? 0 : 1;
}
