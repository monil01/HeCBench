#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void ga(const char *__restrict__ target,
                               const char *__restrict__ query,
                                     char *__restrict__ batch_result,
                                     uint32_t length,
                                     int query_sequence_length,
                                     int coarse_match_length,
                                     int coarse_match_threshold,
                                     int current_position)
{
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > length) return;
  bool match = false;
  int max_length = query_sequence_length - coarse_match_length;

  for (int i = 0; i <= max_length; i++) {
    int distance = 0;
    for (int j = 0; j < coarse_match_length; j++) {
      if (target[current_position + tid + j] != query[i + j]) {
        distance++;
      }
    }

    if (distance < coarse_match_threshold) {
      match = true;
      break;
    }
  }
  if (match) {
    batch_result[tid] = 1;
  }
}
