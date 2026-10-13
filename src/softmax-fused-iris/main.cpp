#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include "reference.h"

#define C10_WARP_SIZE 32
typedef __hip_bfloat16 BFloat16;
typedef __half Half;


#include "dmem.hpp"
#include <type_traits>
int log2_ceil(int value) { int n=0; while((1<<n)<value)++n; return n; }
#define LAUNCH(n) \
  port::launch(std::is_same<input_t, Half>::value ? "fused_half_" #n : "fused_bf_" #n, \
               blocks,threads,dst,src,mask,scale,batch_count,key_seq_len,pad_batches);

template <typename input_t, typename output_t, typename acc_t>
void scaled_masked_softmax_forward(output_t *dst, const input_t *src,
                                   const uint8_t *mask, const acc_t scale,
                                   int query_seq_len, int key_seq_len,
                                   int batches, int attn_heads,
                                   int pad_batches) {
  assert(key_seq_len >= 0 && key_seq_len <= 1024);
  if (key_seq_len == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(key_seq_len);
    const int next_power_of_two = 1 << log2_elements;
    int batch_count = batches * attn_heads * query_seq_len;

    int warp_size =
        (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    assert(query_seq_len % batches_per_block == 0);

    dim3 blocks(query_seq_len / batches_per_block, attn_heads, batches);
    dim3 threads(warp_size, warps_per_block, 1);
    switch (log2_elements) {
    case 0: LAUNCH(0); break;
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break;
    case 4: LAUNCH(4); break;
    case 5: LAUNCH(5); break;
    case 6: LAUNCH(6); break;
    case 7: LAUNCH(7); break;
    case 8: LAUNCH(8); break;
    case 9: LAUNCH(9); break;
    case 10: LAUNCH(10); break;
    default: break;
    }
  }
}

template <typename scalar_t>
void fused_softmax(int batches, int attn_heads, int query_seq_len,
                   int key_seq_len, int pad_batches, int repeat) {
  uint64_t num_data_elems =
      (uint64_t)batches * attn_heads * query_seq_len * key_seq_len;
  float scale_factor = 1.f / sqrtf(key_seq_len);
  uint8_t *mask = (uint8_t*) malloc (sizeof(uint8_t) * num_data_elems);

  srand(123);
  for (size_t i = 0; i < (uint64_t)batches * attn_heads * query_seq_len; i++) {
    uint8_t* mask_row = mask + i * key_seq_len;
    int len = rand() % (key_seq_len / 2);
    memset(mask_row, 1, sizeof(uint8_t) * len); // mask
    memset(mask_row + len, 0, sizeof(uint8_t) * (key_seq_len - len));
  }

  scalar_t *input = make_random<scalar_t>(num_data_elems);

  const int* outliers = make_random_int((uint64_t)batches * attn_heads * query_seq_len, key_seq_len);
  for(int k = 0; k < query_seq_len; ++k) {
    for(int j = 0; j < batches * attn_heads; ++j) {
      input[j * key_seq_len + outliers[j* query_seq_len + k]] *= scalar_t(20);
    }
  }

  scalar_t *output = (scalar_t *)aligned_alloc(1024, sizeof(scalar_t) * num_data_elems);
  scalar_t *output_ref = (scalar_t *)aligned_alloc(1024, sizeof(scalar_t) * num_data_elems);

  scalar_t *d_input, *d_output;
  uint8_t *d_mask;
  port::allocate((void **)&d_input, sizeof(scalar_t) * num_data_elems);
  port::allocate((void **)&d_output, sizeof(scalar_t) * num_data_elems);
  port::allocate((void **)&d_mask, sizeof(uint8_t) * num_data_elems);
  port::copy(d_mask, mask, sizeof(uint8_t) * num_data_elems, port::h2d);
  port::copy(d_input, input, sizeof(scalar_t) * num_data_elems, port::h2d);

  scaled_masked_softmax_forward<scalar_t, scalar_t, float>(
      d_output, d_input, d_mask, scale_factor, query_seq_len, key_seq_len,
      batches, attn_heads, pad_batches);

  port::copy(output, d_output, sizeof(scalar_t) * num_data_elems, port::d2h);

  reference<scalar_t, scalar_t, float>(output_ref, input, mask, scale_factor,
                                       pad_batches, batches, attn_heads, query_seq_len, key_seq_len);
  bool ok = true;
  for (size_t i = 0; i < num_data_elems; i++) {
    if (fabsf((float)output[i] - (float)output_ref[i]) > 1e-3f) {
      printf("Mismatch at index %zu: %f %f\n", i, (float)output[i], (float)output_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  iris_synchronize();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    scaled_masked_softmax_forward<scalar_t, scalar_t, float>(
        d_output, d_input, d_mask, scale_factor, query_seq_len, key_seq_len,
        batches, attn_heads, pad_batches);
  }

  iris_synchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(output);
  free(output_ref);
  free(input);
  free(mask);
  port::release(d_input);
  port::release(d_output);
  port::release(d_mask);
}

int main(int argc, char *argv[]) {
  port::Session session(&argc, &argv);
  if (argc != 6) {
    printf("Usage: %s <batch> <head> <query length> <key length> <repeat>\n",
           argv[0]);
    return 1;
  }

  int batches = atoi(argv[1]);
  int attn_heads = atoi(argv[2]);
  int query_seq_len = atoi(argv[3]);
  int key_seq_len = atoi(argv[4]);
  int repeat = atoi(argv[5]);

  // bert mask
  fused_softmax<Half>(batches, attn_heads, query_seq_len, key_seq_len, 0, repeat);
  // gpt2 mask
  fused_softmax<Half>(batches, attn_heads, query_seq_len, key_seq_len, 1, repeat);
  fused_softmax<BFloat16>(batches, attn_heads, query_seq_len, key_seq_len, 0, repeat);
  fused_softmax<BFloat16>(batches, attn_heads, query_seq_len, key_seq_len, 1, repeat);

  return 0;
}
