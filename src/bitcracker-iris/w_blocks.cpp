#include <chrono>
#include <iostream>
#include "bitcracker.h"
int evaluate_w_block(
    unsigned char * salt,
    iris::DMem &d_w_words_uint32, uint32_t *host_words)
{
  unsigned char * padding;
  uint64_t msgLen;

  if(salt == NULL)
    return BIT_FAILURE;

  padding = (unsigned char *) Calloc(PADDING_SIZE, sizeof(unsigned char));
  padding[0] = 0x80;
  memset(padding + 1, 0, 31);
  msgLen = (FIXED_PART_INPUT_CHAIN_HASH << 3);
  for (int i = 0; i < 8; i++) {
    padding[PADDING_SIZE - 1 - i] = (uint8_t)(msgLen >> (i * 8));
  }

  iris::DMem salt_mem(salt, SALT_SIZE), padding_mem(padding, PADDING_SIZE);
  auto start = std::chrono::steady_clock::now();
  iris::Task task;
  void *args[] = { &salt_mem, &padding_mem, &d_w_words_uint32 };
  int info[] = { iris_r, iris_r, iris_w };
  size_t gws[] = { 1024 * 16 }, lws[] = { 16 };
  iris_check(task.kernel("kernel_w_block", 1, nullptr, gws, lws, 3, args, info));
  iris_check(task.flush_out(d_w_words_uint32));
  iris_check(task.submit(iris_gpu, nullptr, true));
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "kernel time (w_block) : " << time * 1e-3f << " us\n\n";

  free(padding);

  if (getenv("BITCRACKER_VERIFY")) {
    uint64_t digest = 1469598103934665603ULL;
    for (size_t i = 0; i < size_t(NUM_HASH_BLOCKS) * HASH_BLOCK_NUM_UINT32; ++i)
      digest = (digest ^ host_words[i]) * 1099511628211ULL;
    printf("W_DIGEST %016llx\n", (unsigned long long)digest);
  }

  return BIT_SUCCESS;
}
