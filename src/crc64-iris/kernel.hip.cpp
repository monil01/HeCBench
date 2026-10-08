#include <hip/hip_runtime.h>

#include <stddef.h>
#include <stdint.h>

static __device__ const uint64_t crc64_poly = UINT64_C(0xc96c5795d7870f42);

static __device__ uint64_t crc64_device(const unsigned char* input, size_t nbytes,
                                        const uint64_t* table)
{
  const unsigned char* data = input;
  uint64_t cs = UINT64_C(0xffffffffffffffff);

  while (nbytes--) {
    uint32_t idx = static_cast<uint32_t>(cs ^ *data++) & 0xff;
    cs = table[idx] ^ (cs >> 8);
  }

  return cs ^ UINT64_C(0xffffffffffffffff);
}

extern "C" __global__ void crc64_kernel(size_t* d_thread_sz,
                                         uint64_t* d_thread_cs,
                                         const unsigned char* d_data,
                                         const uint64_t* d_crc64_table,
                                         size_t nbytes,
                                         int nthreads)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nthreads) {
    return;
  }

  size_t bpt = nbytes / static_cast<size_t>(nthreads);
  const unsigned char* start = d_data + bpt * static_cast<size_t>(tid);
  const unsigned char* end = (tid != nthreads - 1) ? start + bpt : d_data + nbytes;

  size_t sz = static_cast<size_t>(end - start);
  d_thread_sz[tid] = sz;
  d_thread_cs[tid] = crc64_device(start, sz, d_crc64_table);
}
