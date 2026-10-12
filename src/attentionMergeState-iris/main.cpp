#include <iris/iris.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

static uint32_t xorshift32(uint32_t& state) {
  state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
static float uniform(uint64_t index, float low, float high) {
  uint32_t state = static_cast<uint32_t>(1234 ^ index);
  const float u = (xorshift32(state) >> 8) * 0x1.0p-24f;
  return low + (high - low) * u;
}

static uint16_t float_to_half(float value) {
  uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000u) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
  }
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                ((mantissa + 0x1000u) >> 13));
}
static float half_to_float(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, bits;
  if (!exp) {
    if (!mant) bits = sign;
    else { exp = 127 - 15 + 1; while (!(mant & 0x400u)) { mant <<= 1; --exp; }
      bits = sign | (exp << 23) | ((mant & 0x3ffu) << 13); }
  } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
  else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
static uint16_t float_to_bf16(float value) {
  uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u); return static_cast<uint16_t>(bits >> 16);
}
static float bf16_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16; float result;
  std::memcpy(&result, &bits, sizeof(result)); return result;
}

template <typename T> static float to_float(T value) { return value; }
template <> float to_float<uint16_t>(uint16_t value) { return half_to_float(value); }

template <typename T> static T from_float(float value) { return value; }
template <typename T> static bool is_bf16() { return false; }

struct BFloat16Tag {};
template <> float to_float<BFloat16Tag>(BFloat16Tag) { return 0.0f; }

static bool merge_reference(std::vector<float>& output, std::vector<float>& lse,
                            const std::vector<float>& prefix,
                            const std::vector<float>& suffix,
                            const std::vector<float>& prefix_lse,
                            const std::vector<float>& suffix_lse,
                            uint64_t tokens, uint32_t heads, uint32_t dim) {
  for (uint64_t t = 0; t < tokens; ++t) for (uint32_t h = 0; h < heads; ++h) {
    const uint64_t li = t * heads + h;
    float p = std::isinf(prefix_lse[li]) ? -INFINITY : prefix_lse[li];
    float s = std::isinf(suffix_lse[li]) ? -INFINITY : suffix_lse[li];
    const float m = std::max(p, s), pe = std::exp(p - m), se = std::exp(s - m), sum = pe + se;
    lse[li] = std::log(sum) + m;
    for (uint32_t d = 0; d < dim; ++d) {
      const uint64_t i = li * dim + d;
      output[i] = prefix[i] * (pe / sum) + suffix[i] * (se / sum);
    }
  }
  return true;
}

template <typename T>
static bool run_dtype(const char* name, const char* kernel_name, int32_t repeat,
                      uint64_t tokens, uint32_t heads, uint32_t dim,
                      T (*encode)(float), float (*decode)(T)) {
  const uint64_t elements = tokens * heads * dim, lse_count = tokens * heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  std::vector<T> prefix(elements), suffix(elements), output(elements);
  std::vector<float> prefix_lse(lse_count), suffix_lse(lse_count), lse(lse_count);
  std::vector<float> prefix_ref(elements), suffix_ref(elements), output_ref(elements), lse_ref(lse_count);
  for (uint64_t i = 0; i < elements; ++i) {
    const float p = uniform(i, -scale, scale), s = uniform(i, -scale, scale);
    prefix[i] = encode(p); suffix[i] = encode(s); prefix_ref[i] = decode(prefix[i]); suffix_ref[i] = decode(suffix[i]);
  }
  for (uint64_t i = 0; i < lse_count; ++i) { prefix_lse[i] = uniform(i, -scale, scale); suffix_lse[i] = uniform(i, -scale, scale); }
  merge_reference(output_ref, lse_ref, prefix_ref, suffix_ref, prefix_lse, suffix_lse, tokens, heads, dim);

  iris::DMem mem_output(output.data(), output.size() * sizeof(T));
  iris::DMem mem_prefix(prefix.data(), prefix.size() * sizeof(T));
  iris::DMem mem_suffix(suffix.data(), suffix.size() * sizeof(T));
  iris::DMem mem_lse(lse.data(), lse.size() * sizeof(float));
  iris::DMem mem_prefix_lse(prefix_lse.data(), prefix_lse.size() * sizeof(float));
  iris::DMem mem_suffix_lse(suffix_lse.data(), suffix_lse.size() * sizeof(float));
  const uint32_t pack_size = 16 / sizeof(T), threads_per_head = dim / pack_size;
  const uint64_t total = tokens * heads * threads_per_head;
  size_t gws[1] = {((total + 127) / 128) * 128}, lws[1] = {128};
  void* params[] = {&mem_output, &mem_prefix, &mem_suffix, &mem_lse,
                    &mem_prefix_lse, &mem_suffix_lse, &tokens, &heads, &dim};
  int pinfo[] = {iris_w, iris_r, iris_r, iris_w, iris_r, iris_r,
                 sizeof(tokens), sizeof(heads), sizeof(dim)};
  auto submit = [&]() {
    iris::Task task;
    task.kernel(kernel_name, 1, nullptr, gws, lws, 9, params, pinfo);
    task.flush_out(mem_output); task.flush_out(mem_lse);
    task.submit(iris_gpu, nullptr, 1);
  };
  for (int i = 0; i < 100; ++i) submit();
  bool ok = true;
  for (uint64_t i = 0; i < elements; ++i) if (std::fabs(decode(output[i]) - output_ref[i]) > 1e-3f) ok = false;
  for (uint64_t i = 0; i < lse_count; ++i) if (std::fabs(lse[i] - lse_ref[i]) > 1e-3f) ok = false;
  std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) submit();
  const auto end = std::chrono::steady_clock::now();
  std::printf("Average execution time of the kernel: %f (us)\n",
              std::chrono::duration<double, std::micro>(end - start).count() / repeat);
  return ok;
}

static float identity(float v) { return v; }
static uint16_t encode_half(float v) { return float_to_half(v); }
static uint16_t encode_bf16(float v) { return float_to_bf16(v); }
static float decode_half(uint16_t v) { return half_to_float(v); }
static float decode_bf16(uint16_t v) { return bf16_to_float(v); }

int main(int argc, char** argv) {
  if (argc != 5) { std::printf("Usage: %s <number of tokens> <number of heads> <head size> <repeat>\n", argv[0]); return 1; }
  const uint64_t tokens = std::strtoull(argv[1], nullptr, 10); const uint32_t heads = std::atoi(argv[2]);
  const uint32_t dim = std::atoi(argv[3]); const int32_t repeat = std::atoi(argv[4]);
  if (dim == 0 || dim % 4 != 0) { std::printf("head size must be a positive multiple of 4\n"); return 1; }
  iris::Platform platform; platform.init(&argc, &argv, 1);
  std::printf("#tokens %zu, #heads %u, head dimension %u\n", tokens, heads, dim);
  bool ok = run_dtype<float>("output dtype FP32", "merge_attn_states_fp32", repeat, tokens, heads, dim, identity, identity);
  ok = run_dtype<uint16_t>("output dtype FP16", "merge_attn_states_fp16", repeat, tokens, heads, dim, encode_half, decode_half) && ok;
  ok = run_dtype<uint16_t>("output dtype BF16", "merge_attn_states_bf16", repeat, tokens, heads, dim, encode_bf16, decode_bf16) && ok;
  platform.finalize(); return ok ? 0 : 2;
}
