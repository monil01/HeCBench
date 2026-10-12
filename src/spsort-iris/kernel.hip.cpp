#include <hip/hip_runtime.h>

extern "C" __global__ void csr_sort_rows_float_int(const int* rowptr,
                                                    int* colind,
                                                    float* values,
                                                    int nrows)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) return;

  int begin = rowptr[row];
  int end = rowptr[row + 1];
  for (int i = begin + 1; i < end; i++) {
    int key_col = colind[i];
    float key_val = values[i];
    int j = i - 1;
    while (j >= begin && colind[j] > key_col) {
      colind[j + 1] = colind[j];
      values[j + 1] = values[j];
      j--;
    }
    colind[j + 1] = key_col;
    values[j + 1] = key_val;
  }
}
