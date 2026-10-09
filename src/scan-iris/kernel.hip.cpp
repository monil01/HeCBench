#include <hip/hip_runtime.h>
#include <stdint.h>

#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)

template <typename T, int N>
__device__ void scan_kernel_body(const int64_t nblocks, T* __restrict__ g_odata,
                                 const T* __restrict__ g_idata)
{
  __shared__ T temp[N];
  for (int64_t bid = blockIdx.x; bid < nblocks; bid += gridDim.x) {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;
    int thid = threadIdx.x;
    int offset = 1;
    temp[2 * thid] = gi[2 * thid];
    temp[2 * thid + 1] = gi[2 * thid + 1];
    for (int d = N >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }
    if (thid == 0) temp[N - 1] = 0;
    for (int d = 1; d < N; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        T t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    go[2 * thid] = temp[2 * thid];
    go[2 * thid + 1] = temp[2 * thid + 1];
  }
}

template <typename T, int N>
__device__ void scan_bcao_kernel_body(const int64_t nblocks, T* __restrict__ g_odata,
                                      const T* __restrict__ g_idata)
{
  __shared__ T temp[2 * N];
  for (int64_t bid = blockIdx.x; bid < nblocks; bid += gridDim.x) {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;
    int thid = threadIdx.x;
    int a = thid;
    int b = a + (N / 2);
    int oa = OFFSET(a);
    int ob = OFFSET(b);
    temp[a + oa] = gi[a];
    temp[b + ob] = gi[b];
    int offset = 1;
    for (int d = N >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        ai += OFFSET(ai);
        bi += OFFSET(bi);
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }
    if (thid == 0) temp[N - 1 + OFFSET(N - 1)] = 0;
    for (int d = 1; d < N; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        ai += OFFSET(ai);
        bi += OFFSET(bi);
        T t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    __syncthreads();
    go[a] = temp[a + oa];
    go[b] = temp[b + ob];
  }
}

#define MAKE_SCAN(T, TN, N) \
extern "C" __global__ void scan_##TN##_##N(const int64_t nblocks, T* out, const T* in) { scan_kernel_body<T, N>(nblocks, out, in); } \
extern "C" __global__ void scan_bcao_##TN##_##N(const int64_t nblocks, T* out, const T* in) { scan_bcao_kernel_body<T, N>(nblocks, out, in); }

#define MAKE_ALL(N) \
MAKE_SCAN(char, char, N) \
MAKE_SCAN(short, short, N) \
MAKE_SCAN(int, int, N) \
MAKE_SCAN(long, long, N)

MAKE_ALL(128)
MAKE_ALL(256)
MAKE_ALL(512)
MAKE_ALL(1024)
MAKE_ALL(2048)
