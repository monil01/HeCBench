#include <iris/iris.hpp>

#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

static void log_probs_cpu(float* log_probs,
                          const float* logits,
                          const int* ids,
                          const int* lengths,
                          int max_input_length,
                          int batch_size,
                          int vocab_size,
                          int vocab_size_padded,
                          float* cum_log_probs)
{
  for (int bidx = 0; bidx < batch_size; bidx++) {
    float accum = 0.f;
    for (int step = 0; step < lengths[bidx] - 1; step++) {
      int step_offset = step * vocab_size_padded;
      int batch_offset = bidx * max_input_length * vocab_size_padded;
      const float* logits_ptr = logits + step_offset + batch_offset;
      float max_val = -FLT_MAX;
      for (int i = 0; i < vocab_size; i++) {
        max_val = std::fmax(max_val, logits_ptr[i]);
      }
      float sum_exp = 0.0f;
      for (int i = 0; i < vocab_size; i++) {
        sum_exp += std::exp(logits_ptr[i] - max_val);
      }
      int idx = step + bidx * (max_input_length - 1);
      int token_idx = step + 1 + bidx * max_input_length;
      log_probs[idx] = logits_ptr[ids[token_idx]] - max_val -
                       std::log(sum_exp + 1e-9f);
      accum += log_probs[idx];
    }
    cum_log_probs[bidx] = accum;
  }
}

static void run_kernel(const char* name, int dims, size_t* gws,
                       size_t* lws, int nparams, void** params,
                       int* pinfo, iris::DMem* flush0, iris::DMem* flush1)
{
  iris::Task task;
  task.kernel(name, dims, nullptr, gws, lws, nparams, params, pinfo);
  if (flush0) task.flush_out(*flush0);
  if (flush1) task.flush_out(*flush1);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <maximum sequence length> <batch size> <vocabulary size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int max_length = std::atoi(argv[1]);
  const int batch_size = std::atoi(argv[2]);
  const int vocab_size = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);
  const int vocab_size_padded = (vocab_size + 31) / 32 * 32;

  const size_t logits_size =
      static_cast<size_t>(batch_size) * max_length * vocab_size_padded;
  const size_t logits_size_bytes = logits_size * sizeof(float);
  const size_t log_probs_size = static_cast<size_t>(batch_size) * (max_length - 1);
  const size_t log_probs_size_bytes = log_probs_size * sizeof(float);
  const size_t batch_size_bytes = batch_size * sizeof(float);
  const size_t length_size_bytes = batch_size * sizeof(int);
  const size_t ids_size = static_cast<size_t>(batch_size) * max_length;
  const size_t ids_size_bytes = ids_size * sizeof(int);

  float* h_logits = static_cast<float*>(std::malloc(logits_size_bytes));
  float* h_log_probs = static_cast<float*>(std::malloc(log_probs_size_bytes));
  float* h_log_probs_ref = static_cast<float*>(std::malloc(log_probs_size_bytes));
  float* h_cum_log_probs = static_cast<float*>(std::malloc(batch_size_bytes));
  float* h_cum_log_probs_ref = static_cast<float*>(std::malloc(batch_size_bytes));
  int* h_lengths = static_cast<int*>(std::malloc(length_size_bytes));
  int* h_ids = static_cast<int*>(std::malloc(ids_size_bytes));
  if (!h_logits || !h_log_probs || !h_log_probs_ref || !h_cum_log_probs ||
      !h_cum_log_probs_ref || !h_lengths || !h_ids) {
    std::fprintf(stderr, "host allocation failed\n");
    return 1;
  }

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-6.f, 6.f);
  for (size_t i = 0; i < logits_size; i++) h_logits[i] = distr(g);
  for (int i = 0; i < batch_size; i++) h_lengths[i] = max_length;
  std::srand(123);
  for (size_t i = 0; i < ids_size; i++) h_ids[i] = std::rand() % vocab_size;

  iris::DMem d_logits(h_logits, logits_size_bytes);
  iris::DMem d_log_probs(h_log_probs, log_probs_size_bytes);
  iris::DMem d_cum_log_probs(h_cum_log_probs, batch_size_bytes);
  iris::DMem d_lengths(h_lengths, length_size_bytes);
  iris::DMem d_ids(h_ids, ids_size_bytes);

  const int block_size = vocab_size < 1024 ? (vocab_size + 31) / 32 * 32 : 1024;
  const int gx = max_length - 1;
  const int gy = batch_size;
  size_t log_gws[2] = {static_cast<size_t>(gx * block_size),
                       static_cast<size_t>(gy)};
  size_t log_lws[2] = {static_cast<size_t>(block_size), 1};
  size_t accum_gws[1] = {static_cast<size_t>(batch_size * block_size)};
  size_t accum_lws[1] = {static_cast<size_t>(block_size)};

  void* log_params[8] = {&d_log_probs, &d_logits, &d_ids, &d_lengths,
                         const_cast<int*>(&max_length),
                         const_cast<int*>(&batch_size),
                         const_cast<int*>(&vocab_size),
                         const_cast<int*>(&vocab_size_padded)};
  int log_pinfo[8] = {iris_w, iris_r, iris_r, iris_r,
                      sizeof(max_length), sizeof(batch_size),
                      sizeof(vocab_size), sizeof(vocab_size_padded)};
  void* accum_params[5] = {&d_cum_log_probs, &d_log_probs, &d_lengths,
                           const_cast<int*>(&max_length),
                           const_cast<int*>(&batch_size)};
  int accum_pinfo[5] = {iris_w, iris_r, iris_r, sizeof(max_length),
                        sizeof(batch_size)};

  run_kernel("log_probs_kernel_float", 2, log_gws, log_lws, 8, log_params,
             log_pinfo, &d_log_probs, nullptr);
  run_kernel("accumulate_log_probs", 1, accum_gws, accum_lws, 5, accum_params,
             accum_pinfo, &d_cum_log_probs, nullptr);

  log_probs_cpu(h_log_probs_ref, h_logits, h_ids, h_lengths, max_length,
                batch_size, vocab_size, vocab_size_padded, h_cum_log_probs_ref);
  d_log_probs.fetch(h_log_probs);
  d_cum_log_probs.fetch(h_cum_log_probs);

  bool error = false;
  for (size_t i = 0; i < log_probs_size; i++) {
    if (std::fabs(h_log_probs[i] - h_log_probs_ref[i]) > 1e-3f) {
      std::printf("log_probs: @%zu %f != %f\n", i, h_log_probs[i],
                  h_log_probs_ref[i]);
      error = true;
      break;
    }
  }
  for (int i = 0; i < batch_size; i++) {
    if (std::fabs(h_cum_log_probs[i] - h_cum_log_probs_ref[i]) > 1e-1f) {
      error = true;
      std::printf("cum_log_probs: @%d %f != %f\n", i, h_cum_log_probs[i],
                  h_cum_log_probs_ref[i]);
    }
  }
  std::printf("%s\n", error ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_kernel("log_probs_kernel_float", 2, log_gws, log_lws, 8, log_params,
               log_pinfo, nullptr, nullptr);
    run_kernel("accumulate_log_probs", 1, accum_gws, accum_lws, 5, accum_params,
               accum_pinfo, nullptr, nullptr);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of kernels: %f (us)\n",
              (time * 1e-3f) / repeat);

  std::free(h_cum_log_probs);
  std::free(h_cum_log_probs_ref);
  std::free(h_log_probs);
  std::free(h_log_probs_ref);
  std::free(h_logits);
  std::free(h_ids);
  std::free(h_lengths);
  platform.finalize();
  return 0;
}
