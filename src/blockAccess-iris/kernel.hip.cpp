#include <hip/hip_runtime.h>

#include "block_load.h"
#include "block_store.h"

extern "C" __global__ void reference_kernel(const float * __restrict__ A,
                                             unsigned char *out,
                                             const size_t n)
{
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < n/4; idx += gridDim.x * blockDim.x) {
    const float4 v = reinterpret_cast<const float4*>(A)[idx];
    uchar4 o;
    o.x = (int)v.x;
    o.y = (int)v.y;
    o.z = (int)v.z;
    o.w = (int)v.w;
    reinterpret_cast<uchar4*>(out)[idx] = o;
  }
}

template<int BLOCKSIZE, int ITEMS_PER_THREAD>
__device__ void block_access_kernel_impl(const float * __restrict__ A,
                                         unsigned char *out,
                                         const size_t n)
{
  float vals[ITEMS_PER_THREAD];
  unsigned char qvals[ITEMS_PER_THREAD];

  typedef BlockLoad<float, BLOCKSIZE, ITEMS_PER_THREAD> LoadFloat;
  typedef BlockStore<unsigned char, BLOCKSIZE, ITEMS_PER_THREAD> StoreChar;

  __shared__ typename LoadFloat::TempStorage loadf_storage;
  __shared__ typename StoreChar::TempStorage storec_storage;

  for (size_t i = (size_t)blockIdx.x * BLOCKSIZE * ITEMS_PER_THREAD;
       i < n; i += gridDim.x * BLOCKSIZE * ITEMS_PER_THREAD) {
      int valid_items = min(n - i, (size_t)BLOCKSIZE * ITEMS_PER_THREAD);
      LoadFloat(loadf_storage).Load(&(A[i]), vals, valid_items);

      #pragma unroll
      for(int j = 0; j < ITEMS_PER_THREAD; j++)
          qvals[j] = int(vals[j]);

      StoreChar(storec_storage).Store(&(out[i]), qvals, valid_items);
  }
}

extern "C" __global__ void block_access_kernel(const float * __restrict__ A,
                                               unsigned char *out,
                                               const size_t n)
{
  block_access_kernel_impl<256, 4>(A, out, n);
}
