#include <hip/hip_runtime.h>

extern "C" __global__ void remap_kernel(
    const int* __restrict__ second_order,
    const int* __restrict__ first_order,
    int* __restrict__ output,
    const int n,
    const int k)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= k) return;

  int idx = second_order[i];
  output[first_order[idx]] = i;
  for (idx++; idx < n && (i == k - 1 || idx != second_order[i + 1]); idx++) {
    output[first_order[idx]] = i;
  }
}
