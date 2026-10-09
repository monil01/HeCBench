#include <hip/hip_runtime.h>

extern "C" __global__ void bitonic_sort(const int seq_len,
                                         const int two_power,
                                         int* a)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int seq_num = i / seq_len;
  int swapped_ele = -1;
  int h_len = seq_len / 2;

  if (i < (seq_len * seq_num) + h_len) {
    swapped_ele = i + h_len;
  }

  int odd = seq_num / two_power;
  bool increasing = ((odd % 2) == 0);

  if (swapped_ele != -1) {
    if (((a[i] > a[swapped_ele]) && increasing) ||
        ((a[i] < a[swapped_ele]) && !increasing)) {
      int temp = a[i];
      a[i] = a[swapped_ele];
      a[swapped_ele] = temp;
    }
  }
}
