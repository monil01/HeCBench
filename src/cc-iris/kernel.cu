#include <cuda.h>

static const int ThreadsPerBlock = 256;
static __device__ int topL, posL, topH, posH;

extern "C" __global__ __launch_bounds__(ThreadsPerBlock, 2560 / ThreadsPerBlock)
void cc_init(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  for (int v = from; v < nodes; v += incr) {
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    int m = v;
    int i = beg;
    while ((m == v) && (i < end)) {
      int n = nlist[i];
      m = m < n ? m : n;
      i++;
    }
    nstat[v] = m;
  }
  if (from == 0) { topL = 0; posL = 0; topH = nodes - 1; posH = nodes - 1; }
}

static inline __device__ int representative(const int idx, int* const __restrict__ nstat)
{
  int curr = nstat[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr > (next = nstat[curr])) {
      nstat[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

extern "C" __global__ __launch_bounds__(ThreadsPerBlock, 2560 / ThreadsPerBlock)
void cc_compute1(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, int* const __restrict__ wl)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  for (int v = from; v < nodes; v += incr) {
    const int vstat0 = nstat[v];
    if (v != vstat0) {
      const int beg = nidx[v];
      const int end = nidx[v + 1];
      int deg = end - beg;
      if (deg > 16) {
        int idx = (deg <= 352) ? atomicAdd(&topL, 1) : atomicAdd(&topH, -1);
        wl[idx] = v;
      } else {
        int vstat = representative(v, nstat);
        for (int i = beg; i < end; i++) {
          const int nli = nlist[i];
          if (v > nli) {
            int ostat = representative(nli, nstat);
            bool repeat;
            do {
              repeat = false;
              if (vstat != ostat) {
                int ret;
                if (vstat < ostat) {
                  if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) { ostat = ret; repeat = true; }
                } else {
                  if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) { vstat = ret; repeat = true; }
                }
              }
            } while (repeat);
          }
        }
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(ThreadsPerBlock, 2560 / ThreadsPerBlock)
void cc_compute2(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, int* const __restrict__ wl)
{
  const int lane = threadIdx.x % warpSize;
  int idx;
  if (lane == 0) idx = atomicAdd(&posL, 1);
  idx = __shfl_sync(0xffffffff, idx, 0);
  while (idx < topL) {
    const int v = wl[idx];
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + lane; i < nidx[v + 1]; i += warpSize) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) { ostat = ret; repeat = true; }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) { vstat = ret; repeat = true; }
            }
          }
        } while (repeat);
      }
    }
    if (lane == 0) idx = atomicAdd(&posL, 1);
    idx = __shfl_sync(0xffffffff, idx, 0);
  }
}

extern "C" __global__ __launch_bounds__(ThreadsPerBlock, 2560 / ThreadsPerBlock)
void cc_compute3(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, int* const __restrict__ wl)
{
  __shared__ int vB;
  if (threadIdx.x == 0) {
    const int idx = atomicAdd(&posH, -1);
    vB = (idx > topH) ? wl[idx] : -1;
  }
  __syncthreads();
  while (vB >= 0) {
    const int v = vB;
    __syncthreads();
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + threadIdx.x; i < nidx[v + 1]; i += ThreadsPerBlock) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) { ostat = ret; repeat = true; }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) { vstat = ret; repeat = true; }
            }
          }
        } while (repeat);
      }
    }
    if (threadIdx.x == 0) {
      const int idx = atomicAdd(&posH, -1);
      vB = (idx > topH) ? wl[idx] : -1;
    }
    __syncthreads();
  }
}

extern "C" __global__ __launch_bounds__(ThreadsPerBlock, 2560 / ThreadsPerBlock)
void cc_flatten(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  for (int v = from; v < nodes; v += incr) {
    int next, vstat = nstat[v];
    const int old = vstat;
    while (vstat > (next = nstat[vstat])) {
      vstat = next;
    }
    if (old != vstat) nstat[v] = vstat;
  }
}
