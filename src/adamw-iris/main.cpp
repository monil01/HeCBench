#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>

#include "reference.h"

#define BLOCK_SIZE 64

static void submit_adamw_step(
    iris::DMem& mem_p,
    iris::DMem& mem_g,
    iris::DMem& mem_m_qscale,
    iris::DMem& mem_v_qscale,
    iris::DMem& mem_m,
    iris::DMem& mem_v,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float step,
    long vector_size,
    float correction1,
    float correction2_sqrt,
    float step_size,
    float weight_decay_update,
    float resid_beta1,
    float resid_beta2,
    size_t gws,
    size_t lws)
{
  iris::Task task;
  void* params[19] = {
      &mem_p, &mem_g, &mem_m_qscale, &mem_v_qscale, &mem_m, &mem_v,
      &beta1, &beta2, &lr, &weight_decay, &eps, &step, &vector_size,
      &correction1, &correction2_sqrt, &step_size, &weight_decay_update,
      &resid_beta1, &resid_beta2};
  int pinfo[19] = {
      iris_rw, iris_r, iris_rw, iris_rw, iris_rw, iris_rw,
      sizeof(beta1), sizeof(beta2), sizeof(lr), sizeof(weight_decay), sizeof(eps),
      sizeof(step), sizeof(vector_size), sizeof(correction1), sizeof(correction2_sqrt),
      sizeof(step_size), sizeof(weight_decay_update), sizeof(resid_beta1), sizeof(resid_beta2)};
  size_t gws_arr[1] = {gws};
  size_t lws_arr[1] = {lws};
  task.kernel("fused_4bit_kernel", 1, nullptr, gws_arr, lws_arr, 19, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <vector size> <number of time steps>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const long vector_size = std::atol(argv[1]);
  const int time_step = std::atoi(argv[2]);
  const size_t size_bytes = static_cast<size_t>(vector_size) * 2 * sizeof(float);

  float* g = static_cast<float*>(std::malloc(size_bytes));
  float* p = static_cast<float*>(std::malloc(size_bytes));
  float* p_ref = static_cast<float*>(std::malloc(size_bytes));
  float* m_qscale = static_cast<float*>(std::malloc(size_bytes));
  float* v_qscale = static_cast<float*>(std::malloc(size_bytes));
  int8_t* m = static_cast<int8_t*>(std::malloc(static_cast<size_t>(vector_size)));
  int8_t* v = static_cast<int8_t*>(std::malloc(static_cast<size_t>(vector_size)));
  if (!g || !p || !p_ref || !m_qscale || !v_qscale || !m || !v) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(g);
    std::free(p);
    std::free(p_ref);
    std::free(m_qscale);
    std::free(v_qscale);
    std::free(m);
    std::free(v);
    platform.finalize();
    return 1;
  }

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(0, 1);
  for (long i = 0; i < vector_size * 2; i++) {
    m_qscale[i] = dist(gen);
    v_qscale[i] = dist(gen);
    g[i] = dist(gen);
    p[i] = p_ref[i] = dist(gen);
  }

  for (long i = 0; i < vector_size; i++) {
    m[i] = static_cast<int8_t>(256 * dist(gen));
    v[i] = static_cast<int8_t>(256 * dist(gen));
  }

  iris::DMem mem_m_qscale(m_qscale, size_bytes);
  iris::DMem mem_v_qscale(v_qscale, size_bytes);
  iris::DMem mem_m(m, static_cast<size_t>(vector_size));
  iris::DMem mem_v(v, static_cast<size_t>(vector_size));
  iris::DMem mem_g(g, size_bytes);
  iris::DMem mem_p(p, size_bytes);

  const int threadsPerBlock = BLOCK_SIZE;
  const int blocksPerGrid = (static_cast<int>(vector_size) + threadsPerBlock - 1) / threadsPerBlock;
  const size_t gws = static_cast<size_t>(blocksPerGrid) * threadsPerBlock;
  const size_t lws = threadsPerBlock;

  const float lr = 1e-3f;
  const float weight_decay = 1e-2f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float resid_beta1 = 1.0f - beta1;
  const float resid_beta2 = 1.0f - beta2;
  const float weight_decay_update = 1.0f - lr * weight_decay;

  for (int step = 1; step <= time_step; step++) {
    const float correction1 = 1.0f - std::pow(beta1, step);
    const float correction2_sqrt = std::sqrt(1.0f - std::pow(beta2, step));
    const float step_size = lr / correction1;

    submit_adamw_step(mem_p, mem_g, mem_m_qscale, mem_v_qscale, mem_m, mem_v,
                      beta1, beta2, lr, weight_decay, eps, static_cast<float>(step),
                      vector_size, correction1, correction2_sqrt, step_size,
                      weight_decay_update, resid_beta1, resid_beta2, gws, lws);

    reference<BLOCK_SIZE>(
        blocksPerGrid, p_ref, g, m_qscale, v_qscale, m, v, beta1, beta2, lr,
        weight_decay, eps, static_cast<float>(step), vector_size, correction1,
        correction2_sqrt, step_size, weight_decay_update, resid_beta1, resid_beta2);
  }

  iris::Task verify_flush;
  verify_flush.flush_out(mem_p);
  verify_flush.submit(iris_gpu, nullptr, 1);
  mem_p.fetch(p);

  float absmax_error = 0;
  for (long i = 0; i < vector_size * 2; i++) {
    absmax_error = std::fmax(absmax_error, std::fabs(p[i] - p_ref[i]));
  }
  std::printf("Absolute maximum error: %f\n", absmax_error);
  std::printf("%s\n", absmax_error > 1e-3f ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();
  for (int step = 1; step <= time_step; step++) {
    const float correction1 = 1.0f - std::pow(beta1, step);
    const float correction2_sqrt = std::sqrt(1.0f - std::pow(beta2, step));
    const float step_size = lr / correction1;

    submit_adamw_step(mem_p, mem_g, mem_m_qscale, mem_v_qscale, mem_m, mem_v,
                      beta1, beta2, lr, weight_decay, eps, static_cast<float>(step),
                      vector_size, correction1, correction2_sqrt, step_size,
                      weight_decay_update, resid_beta1, resid_beta2, gws, lws);
  }
  iris::Task timing_flush;
  timing_flush.flush_out(mem_p);
  timing_flush.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (ms)\n", time * 1e-6f / time_step);

  std::free(p);
  std::free(p_ref);
  std::free(m_qscale);
  std::free(v_qscale);
  std::free(m);
  std::free(v);
  std::free(g);
  platform.finalize();
  return absmax_error > 1e-3f ? 1 : 0;
}
