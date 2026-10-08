#include <hip/hip_runtime.h>

static __device__ __forceinline__ int flat_3dim_device(
    int id1, int id2, int id3, int dim2, int dim3)
{
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

extern "C" __global__ void concat(
    const float* __restrict__ inp1,
    const float* __restrict__ inp2,
    float* output,
    int sz0,
    int sz2,
    int sz1_1,
    int sz1_2)
{
  const int nele = sz0 * sz2 * (sz1_1 + sz1_2);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nele) return;

  float* dst_ptr = output + idx;
  const int idx2 = idx % sz2;
  idx = idx / sz2;
  int idx1 = idx % (sz1_1 + sz1_2);
  const int idx0 = idx / (sz1_1 + sz1_2);
  const float* src_ptr;
  int sz1;
  if (idx1 < sz1_1) {
    sz1 = sz1_1;
    src_ptr = inp1;
  } else {
    idx1 -= sz1_1;
    sz1 = sz1_2;
    src_ptr = inp2;
  }
  src_ptr += flat_3dim_device(idx0, idx1, idx2, sz1, sz2);
  *dst_ptr = *src_ptr;
}
