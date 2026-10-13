#include <iris/iris.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static uint32_t xorshift32(uint32_t& state) {
  state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
static float uniform(uint64_t index, float low, float high, uint32_t seed = 1234) {
  uint32_t state = seed ^ static_cast<uint32_t>(index);
  const float u = (xorshift32(state) >> 8) * 0x1.0p-24f;
  return low + (high - low) * u;
}
static float normal(uint64_t index, uint32_t seed = 1234) {
  uint32_t state = seed ^ static_cast<uint32_t>(index);
  const float u1 = (xorshift32(state) >> 8) * 0x1.0p-24f;
  const float u2 = (xorshift32(state) >> 8) * 0x1.0p-24f;
  return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}
static uint16_t float_to_bf16(float value) {
  uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}
static float bf16_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16; float result;
  std::memcpy(&result, &bits, sizeof(result)); return result;
}

template <typename T> static float as_float(T value) { return value; }
template <> float as_float<uint16_t>(uint16_t value) { return bf16_to_float(value); }
template <typename T> static T from_float(float value) { return value; }
template <> uint16_t from_float<uint16_t>(float value) { return float_to_bf16(value); }

template <typename T>
static void reference(T* out, const T* query, const T* key_cache, const T* value_cache,
                     const int* block_tables, const int* seq_lens, const float* slopes,
                     int num_seqs, int num_heads, int num_kv_heads, int head_size,
                     int block_size, int max_blocks, float scale) {
  const int queries_per_kv = num_heads / num_kv_heads;
  for (int s = 0; s < num_seqs; ++s) {
    const int seq_len = seq_lens[s];
    const int* table = block_tables + s * max_blocks;
    for (int h = 0; h < num_heads; ++h) {
      const int kv_h = h / queries_per_kv;
      std::vector<float> logits(seq_len);
      for (int t = 0; t < seq_len; ++t) {
        const int block = table[t / block_size], offset = t % block_size;
        float score = 0.0f;
        for (int d = 0; d < head_size; ++d) {
          const int64_t qi = (static_cast<int64_t>(s) * num_heads + h) * head_size + d;
          const int64_t ki = (static_cast<int64_t>(block) * num_kv_heads + kv_h) * head_size * block_size + d * block_size + offset;
          score += as_float(query[qi]) * as_float(key_cache[ki]);
        }
        score *= scale;
        if (slopes) score += slopes[h] * static_cast<float>(t - seq_len + 1);
        logits[t] = score;
      }
      float max_value = *std::max_element(logits.begin(), logits.end()), sum = 0.0f;
      for (float& value : logits) { value = std::exp(value - max_value); sum += value; }
      for (float& value : logits) value /= sum;
      for (int d = 0; d < head_size; ++d) {
        float result = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
          const int block = table[t / block_size], offset = t % block_size;
          const int64_t vi = (static_cast<int64_t>(block) * num_kv_heads + kv_h) * head_size * block_size + d * block_size + offset;
          result += logits[t] * as_float(value_cache[vi]);
        }
        out[(static_cast<int64_t>(s) * num_heads + h) * head_size + d] = from_float<T>(result);
      }
    }
  }
}

template <typename T>
static bool run_dtype(const char* label, const char* kernel_name, int num_seqs, int num_heads,
                      int head_size, int max_seq_len, int num_blocks, int repeat,
                      uint64_t query_bytes, uint64_t cache_bytes, uint64_t key_bytes,
                      int block_size, float (*decode)(T)) {
  const int64_t query_elems = static_cast<int64_t>(num_seqs) * num_heads * head_size;
  const int max_blocks = (max_seq_len + block_size - 1) / block_size;
  const int64_t cache_elems = static_cast<int64_t>(num_blocks) * num_heads * head_size * block_size;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  std::vector<T> query(query_elems), output(query_elems), reference_output(query_elems);
  std::vector<T> key(cache_elems), value(cache_elems);
  std::vector<int> seq_lens(num_seqs), tables(static_cast<int64_t>(num_seqs) * max_blocks);
  std::vector<float> slopes(num_heads);
  for (int64_t i = 0; i < query_elems; ++i) query[i] = from_float<T>(uniform(i, -scale, scale));
  for (int64_t i = 0; i < cache_elems; ++i) {
    key[i] = from_float<T>(uniform(i, -scale, scale, 1234));
    value[i] = from_float<T>(uniform(i, -scale, scale, 1234));
  }
  for (int i = 0; i < num_heads; ++i) slopes[i] = normal(i);
  std::srand(1234);
  for (int i = 0; i < num_seqs; ++i) seq_lens[i] = std::rand() % max_seq_len + 1;
  for (int64_t i = 0; i < static_cast<int64_t>(tables.size()); ++i) tables[i] = std::rand() % num_blocks;
  reference(reference_output.data(), query.data(), key.data(), value.data(), tables.data(), seq_lens.data(), slopes.data(),
            num_seqs, num_heads, num_heads, head_size, block_size, max_blocks, scale);

  iris::DMem mem_out(output.data(), query_bytes), mem_query(query.data(), query_bytes);
  iris::DMem mem_key(key.data(), key_bytes), mem_value(value.data(), cache_bytes);
  iris::DMem mem_tables(tables.data(), tables.size() * sizeof(int));
  iris::DMem mem_lens(seq_lens.data(), seq_lens.size() * sizeof(int));
  iris::DMem mem_slopes(slopes.data(), slopes.size() * sizeof(float));
  const int zero = 0;
  const size_t gws[2] = {static_cast<size_t>(num_heads) * 64, static_cast<size_t>(num_seqs)}, lws[2] = {64, 1};
  int q_stride = num_heads * head_size;
  int kv_block_stride = num_heads * head_size * block_size;
  int kv_head_stride = head_size * block_size;
  int max_blocks_arg = max_blocks;
  void* params[] = {&mem_out, &mem_query, &mem_key, &mem_value, &num_heads, const_cast<float*>(&scale),
                    &mem_tables, &mem_lens, &max_blocks_arg, &mem_slopes, &q_stride, &kv_block_stride,
                    &kv_head_stride, const_cast<int*>(&zero), const_cast<int*>(&zero), const_cast<int*>(&zero),
                    const_cast<int*>(&zero), const_cast<int*>(&zero)};
  int pinfo[] = {iris_w, iris_r, iris_r, iris_r, sizeof(num_heads), sizeof(scale), iris_r, iris_r,
                 sizeof(max_blocks), iris_r, sizeof(head_size), sizeof(num_blocks), sizeof(num_heads),
                 sizeof(zero), sizeof(zero), sizeof(zero), sizeof(zero), sizeof(zero)};
  auto submit = [&]() {
    iris::Task task;
    task.kernel(kernel_name, 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 18, params, pinfo);
    task.flush_out(mem_out); task.submit(iris_gpu, nullptr, 1);
  };
  for (int i = 0; i < repeat; ++i) submit();
  bool ok = true; float max_error = 0.0f;
  for (int64_t i = 0; i < query_elems; ++i) {
    const float got = decode(output[i]), want = decode(reference_output[i]);
    max_error = std::max(max_error, std::fabs(got - want));
    if (std::fabs(got - want) > 1e-3f + 1e-5f * std::fabs(want)) ok = false;
  }
  std::printf("%s: %s max_abs_error=%g\n", label, ok ? "PASS" : "FAIL", max_error);
  return ok;
}
static float decode_float(float value) { return value; }
static float decode_bf16(uint16_t value) { return bf16_to_float(value); }

int main(int argc, char** argv) {
  if (argc != 7) { std::printf("Usage: %s <batch size> <number of query heads> <head size> <max_seq_len> <number of cache blocks> <repeat>\n", argv[0]); return 1; }
  const int num_seqs = std::atoi(argv[1]), num_heads = std::atoi(argv[2]), head_size = std::atoi(argv[3]);
  const int max_seq_len = std::atoi(argv[4]), num_blocks = std::atoi(argv[5]), repeat = std::atoi(argv[6]);
  if (num_heads <= 0 || head_size != 128 || num_seqs <= 0 || max_seq_len <= 0 || num_blocks <= 0 || repeat <= 0) return 1;
  iris::Platform platform; platform.init(&argc, &argv, 1);
  std::printf("query shape = [%d (num_seqs), %d (num_query_heads), %d (head_size)]\n", num_seqs, num_heads, head_size);
  bool ok = run_dtype<float>("FP32 PageAttention v1", "paged_attention_fp32", num_seqs, num_heads, head_size, max_seq_len, num_blocks, repeat,
                             static_cast<uint64_t>(num_seqs) * num_heads * head_size * sizeof(float),
                             static_cast<uint64_t>(num_blocks) * num_heads * head_size * 16 * sizeof(float),
                             static_cast<uint64_t>(num_blocks) * num_heads * head_size * 16 * sizeof(float), 16, decode_float);
  ok = run_dtype<uint16_t>("BF16 PageAttention v1", "paged_attention_bf16", num_seqs, num_heads, head_size, max_seq_len, num_blocks, repeat,
                           static_cast<uint64_t>(num_seqs) * num_heads * head_size * sizeof(uint16_t),
                           static_cast<uint64_t>(num_blocks) * num_heads * head_size * 32 * sizeof(uint16_t),
                           static_cast<uint64_t>(num_blocks) * num_heads * head_size * 32 * sizeof(uint16_t), 32, decode_bf16) && ok;
  platform.finalize(); return ok ? 0 : 2;
}
