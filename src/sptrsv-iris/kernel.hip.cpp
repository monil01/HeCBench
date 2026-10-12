#include <hip/hip_runtime.h>

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK 8
#endif

__device__ int atomic_load_sptrsv(const int* addr)
{
  const volatile int* vaddr = addr;
  const int value = *vaddr;
  __threadfence();
  return value;
}

__device__ void atomic_store_sptrsv(int* addr, int value)
{
  volatile int* vaddr = addr;
  __threadfence();
  *vaddr = value;
}

extern "C" __global__ void sptrsv_init_buffers(int* get_value,
                                               VALUE_TYPE* x,
                                               const int m,
                                               const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) get_value[i] = 0;
  if (i < n) x[i] = 0;
}

extern "C" __global__ void sptrsv_mix(const int* __restrict__ csrRowPtr,
                                      const int* __restrict__ csrColIdx,
                                      const VALUE_TYPE* __restrict__ csrVal,
                                      int* __restrict__ get_value,
                                      const int m,
                                      const VALUE_TYPE* __restrict__ b,
                                      VALUE_TYPE* __restrict__ x,
                                      const int* __restrict__ warp_num,
                                      const int Len)
{
  const int local_id = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + local_id;
  const int warp_id = global_id / WARP_SIZE;

  __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK * WARP_SIZE];

  if (warp_id >= (Len - 1)) return;

  const int lane_id = (WARP_SIZE - 1) & local_id;

  if (warp_num[warp_id + 1] > (warp_num[warp_id] + 1)) {
    const int row = warp_num[warp_id] + lane_id;
    if (row >= m) return;

    int j = csrRowPtr[row];
    VALUE_TYPE left_sum = 0;
    while (j < csrRowPtr[row + 1]) {
      int col = csrColIdx[j];
      if (atomic_load_sptrsv(&get_value[col]) == 1) {
        left_sum += csrVal[j] * x[col];
        j++;
        col = csrColIdx[j];
      }
      if (row == col) {
        const VALUE_TYPE xi = (b[row] - left_sum) / csrVal[csrRowPtr[row + 1] - 1];
        x[row] = xi;
        atomic_store_sptrsv(&get_value[row], 1);
        j++;
      }
    }
  } else {
    const int row = warp_num[warp_id];
    if (row >= m) return;

    int j = csrRowPtr[row] + lane_id;
    VALUE_TYPE sum = 0;
    while (j < (csrRowPtr[row + 1] - 1)) {
      const int col = csrColIdx[j];
      if (atomic_load_sptrsv(&get_value[col]) == 1) {
        sum += x[col] * csrVal[j];
        j += WARP_SIZE;
      }
    }

    s_left_sum[local_id] = sum;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      if (lane_id < offset) {
        s_left_sum[local_id] += s_left_sum[local_id + offset];
      }
    }

    if (!lane_id) {
      const VALUE_TYPE xi = (b[row] - s_left_sum[local_id]) /
                            csrVal[csrRowPtr[row + 1] - 1];
      x[row] = xi;
      atomic_store_sptrsv(&get_value[row], 1);
    }
  }
}
