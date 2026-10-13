#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#define BLOCK_SIZE 256

static void submit_kernel(const char* name, size_t gx, size_t lx,
                          int n, int d, iris::DMem& key, iris::DMem& query,
                          iris::DMem& dot, iris::DMem& exp_sum,
                          iris::DMem* score, iris::DMem& value,
                          iris::DMem& output, bool flush_output)
{
  iris::Task task;
  size_t gws[1] = {gx};
  size_t lws[1] = {lx};
  void* params[7];
  int pinfo[7];
  int count = 0;

  if (score == nullptr) {
    params[count] = &key; pinfo[count++] = iris_r;
    params[count] = &query; pinfo[count++] = iris_r;
    params[count] = &dot; pinfo[count++] = iris_w;
    params[count] = &exp_sum; pinfo[count++] = iris_rw;
    params[count++] = &n; pinfo[count - 1] = sizeof(n);
    params[count++] = &d; pinfo[count - 1] = sizeof(d);
  } else if (std::strcmp(name, "attention_kernel2") == 0) {
    params[count] = &exp_sum; pinfo[count++] = iris_r;
    params[count] = &dot; pinfo[count++] = iris_r;
    params[count] = score; pinfo[count++] = iris_w;
    params[count++] = &n; pinfo[count - 1] = sizeof(n);
  } else {
    params[count] = &exp_sum; pinfo[count++] = iris_r;
    params[count] = &dot; pinfo[count++] = iris_r;
    params[count] = &value; pinfo[count++] = iris_r;
    params[count] = &output; pinfo[count++] = iris_w;
    params[count++] = &n; pinfo[count - 1] = sizeof(n);
    params[count++] = &d; pinfo[count - 1] = sizeof(d);
  }
  task.kernel(name, 1, nullptr, gws, lws, count, params, pinfo);
  if (flush_output) task.flush_out(output);
  task.submit(iris_gpu, nullptr, 1);
}

static void submit_attention(int impl, int n, int d, iris::DMem& key,
                             iris::DMem& value, iris::DMem& query,
                             iris::DMem& dot, iris::DMem& exp_sum,
                             iris::DMem& score, iris::DMem& output)
{
  (void)dot;
  (void)exp_sum;
  (void)score;
  const char* names[] = {
    "attention_fused_impl0", "attention_fused_impl1",
    "attention_fused_impl2", "attention_fused_impl3"
  };
  iris::Task task;
  size_t gws[1] = {((static_cast<size_t>(d) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE};
  size_t lws[1] = {BLOCK_SIZE};
  void* params[6] = {&key, &value, &query, &output, &n, &d};
  int pinfo[6] = {iris_r, iris_r, iris_r, iris_w, sizeof(n), sizeof(d)};
  task.kernel(names[impl], 1, nullptr, gws, lws, 6, params, pinfo);
  task.flush_out(output);
  task.submit(iris_gpu, nullptr, 1);
}

static void attention_reference(const float* key, const float* value, const float* query,
                               int n, int d, float* output)
{
  float* dot = static_cast<float*>(std::malloc(static_cast<size_t>(n) * sizeof(float)));
  float* score = static_cast<float*>(std::malloc(static_cast<size_t>(n) * sizeof(float)));
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    dot[i] = 0.0f;
    for (int j = 0; j < d; ++j) dot[i] += key[i * d + j] * query[j];
    sum += std::exp(dot[i]);
  }
  for (int i = 0; i < n; ++i) score[i] = std::exp(dot[i]) / sum;
  for (int j = 0; j < d; ++j) {
    output[j] = 0.0f;
    for (int i = 0; i < n; ++i) output[j] += score[i] * value[i * d + j];
  }
  std::free(dot);
  std::free(score);
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    std::printf("Usage: %s <rows> <columns> <implementation> <repeat>\n", argv[0]);
    std::printf("implementation 0: naive\nimplementation 1: fused kernels with block reduce\n");
    std::printf("implementation 2: fused kernels with warp reduce\nimplementation 3: fused kernels with mixed reduce\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  const int n = std::atoi(argv[1]);
  const int d = std::atoi(argv[2]);
  const int impl = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);
  if (n <= 0 || d <= 0 || repeat <= 0 || impl < 0 || impl > 3) {
    std::fprintf(stderr, "invalid attention arguments\n");
    platform.finalize();
    return 1;
  }

  const size_t kv_count = static_cast<size_t>(n) * d;
  const size_t kv_bytes = kv_count * sizeof(float);
  const size_t q_bytes = static_cast<size_t>(d) * sizeof(float);
  const size_t r_bytes = static_cast<size_t>(n) * sizeof(float);
  float* key = static_cast<float*>(std::malloc(kv_bytes));
  float* value = static_cast<float*>(std::malloc(kv_bytes));
  float* query = static_cast<float*>(std::malloc(q_bytes));
  float* output = static_cast<float*>(std::malloc(q_bytes));
  float* expected = static_cast<float*>(std::malloc(q_bytes));
  if (!key || !value || !query || !output || !expected) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
  for (size_t i = 0; i < kv_count; ++i) {
    key[i] = dist(gen);
    value[i] = dist(gen);
    query[i % static_cast<size_t>(d)] = dist(gen);
  }
  attention_reference(key, value, query, n, d, expected);

  float zero = 0.0f;
  iris::DMem mem_key(key, kv_bytes);
  iris::DMem mem_value(value, kv_bytes);
  iris::DMem mem_query(query, q_bytes);
  float* dot = static_cast<float*>(std::calloc(static_cast<size_t>(n), sizeof(float)));
  float* score = static_cast<float*>(std::calloc(static_cast<size_t>(n), sizeof(float)));
  if (!dot || !score) {
    std::fprintf(stderr, "intermediate allocation failed\n");
    std::free(key); std::free(value); std::free(query); std::free(output); std::free(expected);
    std::free(dot); std::free(score);
    platform.finalize();
    return 1;
  }
  iris::DMem mem_dot(dot, r_bytes);
  iris::DMem mem_exp_sum(&zero, sizeof(float));
  iris::DMem mem_score(score, r_bytes);
  iris::DMem mem_output(output, q_bytes);

  const auto start = std::chrono::steady_clock::now();
  for (int k = 0; k < repeat; ++k) {
    mem_exp_sum.update(&zero);
    submit_attention(impl, n, d, mem_key, mem_value, mem_query, mem_dot,
                     mem_exp_sum, mem_score, mem_output);
  }
  mem_output.fetch(output);
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of kernels %f (ms)\n", elapsed * 1.0e-6 / repeat);

  bool ok = true;
  float max_error = 0.0f;
  for (int i = 0; i < d; ++i) {
    max_error = std::fmax(max_error, std::fabs(expected[i] - output[i]));
    if (std::fabs(expected[i] - output[i]) > 1.0e-3f) ok = false;
  }
  std::printf("max_abs_error %.9g\n", max_error);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(key); std::free(value); std::free(query); std::free(output); std::free(expected);
  std::free(dot); std::free(score);
  platform.finalize();
  return ok ? 0 : 1;
}
