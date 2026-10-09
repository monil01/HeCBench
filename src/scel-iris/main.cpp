#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "reference.h"

#define GPU_NUM_THREADS 256

static void SubmitScel(int inner_size, int log_D_trick, int unjoined_lr_loss,
                       iris::DMem& logits_mem, iris::DMem& targets_mem,
                       iris::DMem& out_mem, int outer_size, bool flush)
{
  size_t gws[1] = {static_cast<size_t>(outer_size) * GPU_NUM_THREADS};
  size_t lws[1] = {GPU_NUM_THREADS};
  iris::Task task;
  void* params[6] = {&inner_size, &log_D_trick, &unjoined_lr_loss,
                     &logits_mem, &targets_mem, &out_mem};
  int pinfo[6] = {sizeof(inner_size), sizeof(log_D_trick), sizeof(unjoined_lr_loss),
                  iris_r, iris_r, iris_w};
  task.kernel("SigmoidCrossEntropyWithLogitsKernel", 1, nullptr, gws, lws,
              6, params, pinfo);
  if (flush) task.flush_out(out_mem);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <outer size> <inner_size> <repeat>\n", argv[0]);
    return 1;
  }

  const int outer_size = std::atoi(argv[1]);
  const int inner_size = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  if (outer_size <= 0 || inner_size <= 0 || repeat <= 0) {
    std::fprintf(stderr, "arguments must be positive\n");
    return 1;
  }

  int input_size = (outer_size + 1) * inner_size;
  size_t input_size_bytes = static_cast<size_t>(input_size) * sizeof(float);
  int output_size = outer_size;
  size_t output_size_bytes = static_cast<size_t>(output_size) * sizeof(float);

  std::default_random_engine generator(123);
  std::normal_distribution<float> distribution(0, 1);

  float* h_logits = static_cast<float*>(std::malloc(input_size_bytes));
  float* h_targets = static_cast<float*>(std::malloc(input_size_bytes));
  float* h_out = static_cast<float*>(std::malloc(output_size_bytes));
  float* r_out = static_cast<float*>(std::malloc(output_size_bytes));
  if (h_logits == nullptr || h_targets == nullptr || h_out == nullptr || r_out == nullptr) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_logits);
    std::free(h_targets);
    std::free(h_out);
    std::free(r_out);
    return 1;
  }

  for (int i = 0; i < input_size; i++) {
    h_logits[i] = distribution(generator);
    h_targets[i] = distribution(generator) + 1.f;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem logits_mem(h_logits, input_size_bytes);
  iris::DMem targets_mem(h_targets, input_size_bytes);
  iris::DMem out_mem(h_out, output_size_bytes);
  logits_mem.update(h_logits);
  targets_mem.update(h_targets);

  bool ok = true;
  for (int unjoined_lr_loss = 0; unjoined_lr_loss <= 1; unjoined_lr_loss++) {
    int logD = (unjoined_lr_loss == 0) ? 1 : 0;

    for (int logD_trick = 0; logD_trick <= logD; logD_trick++) {
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++) {
        SubmitScel(inner_size, logD_trick, unjoined_lr_loss,
                   logits_mem, targets_mem, out_mem, outer_size, i == repeat - 1);
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      std::printf("Average execution time of SigmoidCrossEntropyWithLogits kernel: %f (us)\n",
                  (time * 1e-3f) / repeat);

      out_mem.fetch(h_out);
      reference(outer_size, inner_size, logD_trick, unjoined_lr_loss,
                h_logits, h_targets, r_out);
      for (int i = 0; i < output_size; i++) {
        if (std::fabs(r_out[i] - h_out[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  std::free(h_targets);
  std::free(h_logits);
  std::free(h_out);
  std::free(r_out);
  return 0;
}
