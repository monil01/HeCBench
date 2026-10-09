#include <hip/hip_runtime.h>

extern "C" __global__ void segment_reduce_kernel(int* out,
                                                  const int* in,
                                                  int num_segments,
                                                  int segment_size)
{
  const int segment = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (segment >= num_segments) return;

  const int begin = segment * segment_size;
  int sum = 0;
  for (int i = 0; i < segment_size; ++i) {
    sum += in[begin + i];
  }
  out[segment] = sum;
}
