#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "../addBiasQKV-cuda/reference.h"

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int batch_size = 8;
  const int seq_len = 1024;
  int hidden_units = 768;
  int head_num = 12;
  const int repeat = 1000;
  int valid_word_num = batch_size * seq_len;
  int size_per_head = hidden_units / head_num;
  const int bias_size = 3 * hidden_units;
  const int total = valid_word_num * 3 * hidden_units;
#if IRIS_FP8_FNUZ
  const bool fnuz = true;
#else
  const bool fnuz = false;
#endif

  std::vector<uint8_t> h_src(total);
  std::vector<uint16_t> h_bias(bias_size);
  std::vector<uint8_t> h_out(total, 0);
  std::vector<uint8_t> h_ref(total, 0);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (int i = 0; i < total; ++i)
    h_src[i] = ref::f32_to_fp8_e4m3(dist(rng), fnuz);
  for (int i = 0; i < bias_size; ++i)
    h_bias[i] = ref::f32_to_bf16(dist(rng));

  float input_scale = 0.5f;
  float output_scale = 1.25f;

  iris::DMem mem_out(h_out.data(), h_out.size() * sizeof(uint8_t));
  iris::DMem mem_src(h_src.data(), h_src.size() * sizeof(uint8_t));
  iris::DMem mem_bias(h_bias.data(), h_bias.size() * sizeof(uint16_t));
  iris::DMem mem_input_scale(&input_scale, sizeof(input_scale));
  iris::DMem mem_output_scale(&output_scale, sizeof(output_scale));

  void* params[] = {&mem_out, &mem_src, &mem_bias, &mem_input_scale,
                    &mem_output_scale, &valid_word_num, &head_num,
                    &size_per_head, &hidden_units};
  int pinfo[] = {iris_w, iris_r, iris_r, iris_r, iris_r, sizeof(valid_word_num),
                 sizeof(head_num), sizeof(size_per_head), sizeof(hidden_units)};
  size_t gws[] = {static_cast<size_t>(valid_word_num),
                  static_cast<size_t>(size_per_head / 4),
                  static_cast<size_t>(head_num * 3)};
  size_t lws[] = {1, 1, 1};

  iris::Task task("addBiasQKV", true);
  task.kernel("FP8TrtAddQKVBiasKernel", 3, nullptr, gws, lws, 9, params, pinfo);
  task.flush_out(mem_out);

  const auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; ++i)
    task.submit(iris_gpu, nullptr, true);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the kernel: %f (us)\n",
              (elapsed_ns * 1e-3f) / repeat);

  addBiasQKV_reference(h_ref.data(), h_src.data(), h_bias.data(), input_scale,
                       output_scale, valid_word_num, head_num, size_per_head,
                       hidden_units, fnuz);

  int mismatches = 0;
  for (int i = 0; i < total; ++i)
    if (h_out[i] != h_ref[i]) ++mismatches;
  std::printf("%s\n", mismatches == 0 ? "PASS" : "FAIL");

  platform.finalize();
  return mismatches == 0 ? 0 : 1;
}
