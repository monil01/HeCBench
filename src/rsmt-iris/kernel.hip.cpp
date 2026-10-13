#include <hip/hip_runtime.h>
#include <climits>
#include "types.h"


template <int WarpsPerBlock, int PinLimit>
static __device__
void buildMST(const ID num,
              const coord_t* const __restrict__ x,
              const coord_t* const __restrict__ y,
              edge* const __restrict__ edges,
              coord_t dist[PinLimit])
{
  const int WS = warpSize;
  __shared__ ID source[WarpsPerBlock][PinLimit];
  __shared__ ID destin[WarpsPerBlock][PinLimit];
  __shared__ coord_t mindj[WarpsPerBlock];

  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;

  // initialize
  ID numItems = num - 1;
  for (ID i = lane; i < numItems; i += WS) dist[i] = INT_MAX;  // change if coord_t changed
  for (ID i = lane; i < numItems; i += WS) destin[warp][i] = (ID)(i + 1);

  // Prim's MST algorithm
  ID src = 0;
  for (ID cnt = 0; cnt < num - 1; cnt++) {
    if (lane == 0) mindj[warp] = INT_MAX;

    // update distances
    for (ID j = lane; j < numItems; j += WS) {
      const ID dst = destin[warp][j];
      const coord_t dnew = abs(x[src] - x[dst]) + abs(y[src] - y[dst]);
      coord_t d = dist[j];
      if (d > dnew) {
        d = dnew;
        dist[j] = dnew;
        source[warp][j] = src;
      }
      const int upv = d * (MaxPins * 2) + j;  // tie breaker for determinism
      atomicMin((coord_t*)&mindj[warp], upv);
    }

    // create new edge
    const ID j = mindj[warp] % (MaxPins * 2);
    src = destin[warp][j];
    numItems--;
    if (lane == 0) {
      edges[cnt].src = source[warp][j];
      edges[cnt].dst = src;
      dist[j] = dist[numItems];
      source[warp][j] = source[warp][numItems];
      destin[warp][j] = destin[warp][numItems];
    }
  }
}

template <int WarpsPerBlock, int PinLimit>
static __device__
bool insertSteinerPoints(ID& num,
                         coord_t* const __restrict__ x,
                         coord_t* const __restrict__ y,
                         const edge* const __restrict__ edges,
                         coord_t dist[PinLimit])
{
  const int WS = warpSize;
  __shared__ ID adj[WarpsPerBlock][PinLimit][8];
  __shared__ int cnt[WarpsPerBlock][PinLimit];

  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const ID top = num;

  // create adjacency lists
  for (ID i = lane; i < top; i += WS) cnt[warp][i] = 0;
  for (ID e = lane; e < top - 1; e += WS) {
    dist[e] = -1;
    const ID s = edges[e].src;
    const ID d = edges[e].dst;
    if ((x[d] != x[s]) || (y[d] != y[s])) {
      const int ps = atomicAdd(&cnt[warp][s], 1);
      adj[warp][s][ps] = e;
      const int pd = atomicAdd(&cnt[warp][d], 1);
      adj[warp][d][pd] = e;
    }
  }

  // find best distance for each triangle
  for (ID s = lane; s < top; s += WS) {
    if (cnt[warp][s] >= 2) {
      const coord_t x0 = x[s];
      const coord_t y0 = y[s];
      for (char j = 0; j < cnt[warp][s] - 1; j++) {
        const ID e1 = adj[warp][s][j];
        const ID d1 = (s != edges[e1].src) ? edges[e1].src : edges[e1].dst;
        const coord_t x1 = x[d1];
        const coord_t y1 = y[d1];
        for (char k = j + 1; k < cnt[warp][s]; k++) {
          const ID e2 = adj[warp][s][k];
          const ID d2 = (s != edges[e2].src) ? edges[e2].src : edges[e2].dst;
          const coord_t stx = max(min(x0, x1), min(max(x0, x1), x[d2]));
          const coord_t sty = max(min(y0, y1), min(max(y0, y1), y[d2]));
          const coord_t rd = abs(stx - x0) + abs(sty - y0);
          if (rd > 0) {
            const coord_t rd1 = rd * (MaxPins * 2) + e1;  // tie breaker
            const coord_t rd2 = rd * (MaxPins * 2) + e2;  // tie breaker
            atomicMax((coord_t*)&dist[e1], rd2);
            atomicMax((coord_t*)&dist[e2], rd1);
          }
        }
      }
    }
  }

  // process "triangles" to find best candidate Steiner points
  bool updated = false;
  for (ID e1 = lane; __any(e1 < top - 2); e1 += WS) {
    bool insert = false;
    coord_t stx, sty;
    if (e1 < top - 2) {
      const coord_t d1 = dist[e1];
      if (d1 > 0) {
        const ID e2 = d1 % (MaxPins * 2);
        if (e2 > e1) {
          const coord_t d2 = dist[e2];
          if (e1 == d2 % (MaxPins * 2)) {
            const coord_t x0 = x[edges[e1].src];
            const coord_t y0 = y[edges[e1].src];
            const coord_t x1 = x[edges[e1].dst];
            const coord_t y1 = y[edges[e1].dst];
            coord_t x2 = x[edges[e2].src];
            coord_t y2 = y[edges[e2].src];
            if (((x2 == x0) && (y2 == y0)) || ((x2 == x1) && (y2 == y1))) {
              x2 = x[edges[e2].dst];
              y2 = y[edges[e2].dst];
            }
            updated = true;
            insert = true;
            stx = max(min(x0, x1), min(max(x0, x1), x2));
            sty = max(min(y0, y1), min(max(y0, y1), y2));
          }
        }
      }
    }
    const unsigned long long bal = __ballot(insert);
    const int pos = __popcll(bal & ~(-1ULL << lane)) + num;
    if (insert) {
      x[pos] = stx;
      y[pos] = sty;
    }
    num += __popcll(bal);
  }

  return __any(updated);
}

template <int WarpsPerBlock, int PinLimit>
static __device__
inline void processSmallNet(const int i,
                            const int* const __restrict__ idxin,
                            const coord_t* const __restrict__ xin,
                            const coord_t* const __restrict__ yin,
                            int* const __restrict__ idxout,
                            coord_t* const __restrict__ xout,
                            coord_t* const __restrict__ yout,
                             edge* const __restrict__ edges,
                              int* const __restrict__ wl, int* const __restrict__ counters)
{
  const int WS = warpSize;
  __shared__ coord_t dist[WarpsPerBlock][PinLimit];
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;

  // initialize arrays and copy input coords to output
  const int pin = idxin[i];
  const ID num = idxin[i + 1] - pin;
  const int pout = 2 * pin;
  if (lane == 0) idxout[i] = pout;
  for (ID j = lane; j < num; j += WS) xout[pout + j] = xin[pin + j];
  for (ID j = lane; j < num; j += WS) yout[pout + j] = yin[pin + j];

  // process nets
  if (num == 2) {
    if (lane == 0) edges[pout] = edge{0, 1};
  } else if (num == 3) {
    coord_t x0, y0;
    if (lane < 3) {
      edges[pout + lane] = edge{(short)lane, 3};
      x0 = xout[pout + lane];
      y0 = yout[pout + lane];
    }
    const coord_t x1 = __shfl(x0, 1);
    const coord_t y1 = __shfl(y0, 1);
    const coord_t x2 = __shfl(x0, 2);
    const coord_t y2 = __shfl(y0, 2);
    if (lane == 0) {
      xout[pout + 3] = max(min(x0, x1), min(max(x0, x1), x2));
      yout[pout + 3] = max(min(y0, y1), min(max(y0, y1), y2));
    }
  } else if (num <= 32) {
    // iterate until all Steiner points added
    ID cnt = num;
    do {
      buildMST<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp]);
    } while (insertSteinerPoints<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp]));
  } else {
    if (lane == 0) wl[atomicAdd(&counters[2], 1)] = i;
  }
}

template <int WarpsPerBlock, int PinLimit>
static __device__
inline void processLargeNet(const int i,
                            const int* const __restrict__ idxin,
                            coord_t* const __restrict__ xout,
                            coord_t* const __restrict__ yout,
                             edge* const __restrict__ edges)
{
  const int WS = warpSize;
  __shared__ coord_t dist[WarpsPerBlock][PinLimit];
  const int warp = threadIdx.x / WS;

  const int pin = idxin[i];
  const ID num = idxin[i + 1] - pin;
  const int pout = 2 * pin;

  // iterate until all Steiner points added
  ID cnt = num;
  do {
    buildMST<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp]);
  } while (insertSteinerPoints<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp]));
}

extern "C" __global__ __launch_bounds__(768, 2)
void largeNetKernel(const int* const __restrict__ idxin,
                    const coord_t* const __restrict__ xin,
                    const coord_t* const __restrict__ yin,
                    int* const __restrict__ idxout,
                    coord_t* __restrict__ xout,
                    coord_t* __restrict__ yout,
                     edge* __restrict__ edges,
                    const int numnets,
                    int* const __restrict__ wl, int* const __restrict__ counters)
{
  const int WS = warpSize;
  // compute Steiner points and edges
  const int lane = threadIdx.x % WS;
  do {
    int i;
    if (lane == 0) i = atomicAdd(&counters[0], 1);
    i = __shfl(i, 0);
    if (i >= numnets) break;
    processSmallNet<12, 64>(i, idxin, xin, yin, idxout, xout, yout, edges, wl, counters);
  } while (true);

  // set final element
  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
    idxout[numnets] = 2 * idxin[numnets];
  }
}

extern "C" __global__ __launch_bounds__(192, 2)
void smallNetKernel(const int* const __restrict__ idxin,
                    coord_t* __restrict__ xout,
                    coord_t* __restrict__ yout,
                     edge* __restrict__ edges,
                      int* const __restrict__ wl, int* const __restrict__ counters)
{
  const int WS = warpSize;
  // compute Steiner points and edges
  const int lane = threadIdx.x % WS;
  do {
    int i;
    if (lane == 0) i = atomicAdd(&counters[1], 1);
    i = __shfl(i, 0);
    if (i >= counters[2]) break;
    processLargeNet<3, 512>(wl[i], idxin, xout, yout, edges);
  } while (true);
}


extern "C" __global__ void waveInfo(int* out) {
  if (threadIdx.x == 0) out[0] = warpSize;
}
