#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "reference.h"

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <target sequence length> <query sequence length> "
                "<coarse match length> <coarse match threshold>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int kBatchSize = 1024;
  char seq[] = {'A', 'C', 'T', 'G'};
  const int tseq_size = std::atoi(argv[1]);
  const int qseq_size = std::atoi(argv[2]);
  const int coarse_match_length = std::atoi(argv[3]);
  const int coarse_match_threshold = std::atoi(argv[4]);

  std::vector<char> target_sequence(tseq_size);
  std::vector<char> query_sequence(qseq_size);

  std::srand(123);
  for (int i = 0; i < tseq_size; i++) target_sequence[i] = seq[std::rand() % 4];
  for (int i = 0; i < qseq_size; i++) query_sequence[i] = seq[std::rand() % 4];

  iris::DMem d_target(target_sequence.data(), (size_t)tseq_size * sizeof(char));
  iris::DMem d_query(query_sequence.data(), (size_t)qseq_size * sizeof(char));

  uint32_t max_searchable_length = (uint32_t)(tseq_size - coarse_match_length);
  uint32_t current_position = 0;

  char batch_result[kBatchSize];
  char batch_result_ref[kBatchSize];
  iris::DMem d_batch_result(batch_result, kBatchSize * sizeof(char));

  float total_time = 0.f;

  int error = 0;
  while (current_position < max_searchable_length) {
    std::memset(batch_result, 0, kBatchSize);
    std::memset(batch_result_ref, 0, kBatchSize);
    d_batch_result.update(batch_result);

    uint32_t end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    uint32_t length = end_position - current_position;

    size_t lws[1] = {256u};
    size_t gws[1] = {((size_t)length + lws[0] - 1u) / lws[0] * lws[0]};

    auto start = std::chrono::steady_clock::now();
    iris::Task task;
    void* params[8] = {
      &d_target,
      &d_query,
      &d_batch_result,
      &length,
      (void*)&qseq_size,
      (void*)&coarse_match_length,
      (void*)&coarse_match_threshold,
      &current_position
    };
    int pinfo[8] = {
      iris_r,
      iris_r,
      iris_rw,
      (int)sizeof(length),
      (int)sizeof(qseq_size),
      (int)sizeof(coarse_match_length),
      (int)sizeof(coarse_match_threshold),
      (int)sizeof(current_position)
    };
    task.kernel("ga", 1, nullptr, gws, lws, 8, params, pinfo);
    task.flush_out(d_batch_result);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    reference(target_sequence.data(), query_sequence.data(), batch_result_ref, length, qseq_size,
              coarse_match_length, coarse_match_threshold, (int)current_position);

    error = std::memcmp(batch_result_ref, batch_result, kBatchSize * sizeof(char));
    if (error) break;

    current_position = end_position;
  }
  std::printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);
  std::printf("%s\n", error ? "FAIL" : "PASS");

  platform.finalize();
  return error ? 2 : 0;
}
