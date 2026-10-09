#include <cuda_runtime.h>

extern "C" __global__ void wordcount_mark_starts(unsigned char* starts,
                                                  const char* input,
                                                  size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const char right = input[i];
  const bool right_alpha = right >= 'A' && right <= 'z';
  bool left_alpha = false;
  if (i > 0) {
    const char left = input[i - 1];
    left_alpha = left >= 'A' && left <= 'z';
  }
  starts[i] = (right_alpha && !left_alpha) ? 1 : 0;
}
