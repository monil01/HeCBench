/*
   Kernels for softmax forward pass.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <hipcub/hipcub.hpp>
#include <hip/hip_runtime.h>
#include <cmath>
#include "reference.h"

// online softmax paper: http://arxiv.org/abs/1805.02867
// online softmax reduces loops from 3 to 2
// which is done by calculating sumval and maxval in one loop

/* struct for the reduction operation, guarantees 8-byte alignment
struct __align__(8) SumMax
{
    float maxval;
    float sum;
};

// forceinline helps avoid function call overhead
__device__ __forceinline__ SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger_m = a_bigger ? a : b;
    SumMax smaller_m = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger_m.maxval;
    res.sum = bigger_m.sum + smaller_m.sum * expf(smaller_m.maxval - bigger_m.maxval);
    return res;
}

__device__ void softmax_forward_online_kernel(float* out, const float* inp, int N, int C) {
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  int row = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (row >= N) {
    return;
  }

  // one row of inp, i.e. inp[row, :] of shape (C,)
  const float* x = inp + row * C;
  float* const y = out + row * C;

  // base case for the reduction
  SumMax sm_partial;
  sm_partial.maxval = -INFINITY;
  sm_partial.sum = 0.0f;

  // first, thread coarsening by directly accessing global memory in series
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    sm_partial = reduce_sum_max_op(sm_partial, { x[i], 1.0f });
  }

  // second, the reduction
  SumMax sm_total = cg::reduce(warp, sm_partial, reduce_sum_max_op);

  // divide the whole row by the sum
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    y[i] = expf(x[i] - sm_total.maxval) / sm_total.sum;
  }
}
*/

template <unsigned int WarpSize>
__device__ void softmax_forward_online_kernel2(float* out, const float* inp, int N, int C) {
  int tid = threadIdx.x;
  if (tid >= C) return;

  int warpId = tid / WarpSize;
  const int warpsPerBlock = blockDim.x / WarpSize;
  int row = blockIdx.x * warpsPerBlock + warpId;

  if (row >= N) return;

  int laneId = tid % WarpSize;
  const float* x = inp + row * C;
  float* const y = out + row * C;

  // merge calculating maxval and sumval in one loop
  // which is an arithmetic improvment from online softmax over normal softmax
  float maxval = -INFINITY, sumval = 0.0f, bigger;
  for (int i = laneId; i < C; i += WarpSize) {
    // when updating the maxval, dynamically updates the previous sumval by
    // multiplying e^{previous_maxval - current_maxval}
    bigger = fmaxf(maxval, x[i]);
    sumval = sumval * expf(maxval - bigger) + expf(x[i] - bigger);
    maxval = bigger;
  }

  // use warp functions instead of cooperative groups for better readibility
  // calculate the warp wised maxval and sumval
  float offsetMaxval, offsetSumval;
  for (int offset = WarpSize / 2; offset > 0; offset >>= 1) {
    offsetMaxval = __shfl_down(maxval, offset, WarpSize);
    offsetSumval = __shfl_down(sumval, offset, WarpSize);
    if (offsetMaxval > maxval) {
      sumval *= expf(maxval - offsetMaxval);
      maxval = offsetMaxval;
    } else {
      offsetSumval *= expf(offsetMaxval - maxval);
    }
    sumval += offsetSumval;
  }

  // sync the warp wised maxval and sumval
  // which are also the maxval and sumval of one row in C
  maxval = __shfl(maxval, 0, WarpSize);
  sumval = __shfl(sumval, 0, WarpSize);

  for (int i = laneId; i < C; i += WarpSize) {
    y[i] = expf(x[i] - maxval) / sumval;
  }
}

__device__ void softmax_forward_online_kernel3(float* __restrict__ out, const float* __restrict__ inp, int N, int C) {
  __shared__ float smem[1024];

  int row = blockIdx.x;
  if (row >= N) return;

  const float* x = inp + row * C;
  float* y = out + row * C;
  float maxval = -INFINITY;
  float sumval = 0.0f;

  int tid = threadIdx.x;
  for (int i = tid; i < C; i += blockDim.x) {
      float v = x[i];
      if (v > maxval) {
          sumval *= expf(maxval - v);
          maxval = v;
      }
      sumval += expf(v - maxval);
  }

  smem[tid] = maxval;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
          smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
      }
      __syncthreads();
  }

  float row_max = smem[0];
  __syncthreads();

  smem[tid] = sumval * expf(maxval - row_max);
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
          smem[tid] += smem[tid + stride];
      }
      __syncthreads();
  }
  float row_sum = smem[0];
  __syncthreads();

  for (int i = tid; i < C; i += blockDim.x) {
      y[i] = expf(x[i] - row_max) / row_sum;
  }
}

// Baseline softmax
template <unsigned int WarpSize>
__device__ void softmax_forward_baseline_kernel(float* out, const float* inp, int N, int C) {
  int tid = threadIdx.x;
  if (tid >= C) return;

  int warpId = tid / WarpSize;
  const int warpsPerBlock = blockDim.x / WarpSize;
  int row = blockIdx.x * warpsPerBlock + warpId;

  if (row >= N) return;

  int laneId = tid % WarpSize;
  const float* x = inp + row * C;
  float* const y = out + row * C;

  using WarpReduce = hipcub::WarpReduce<float, WarpSize>;
  __shared__ typename WarpReduce::TempStorage temp[1024/WarpSize];

  float maxval = -INFINITY;
  for (int i = laneId; i < C; i += WarpSize) {
    maxval = max(x[i], maxval);
  }
  maxval = WarpReduce(temp[warpId]).Reduce(maxval, hipcub::Max());

  maxval = __shfl(maxval, 0, WarpSize);

  float sumval = 0;
  for (int i = laneId; i < C; i += WarpSize) {
    sumval += expf(x[i] - maxval);
  }
  sumval = WarpReduce(temp[warpId]).Sum(sumval);

  sumval = __shfl(sumval, 0, WarpSize);

  for (int i = laneId; i < C; i += WarpSize) {
    y[i] = expf(x[i] - maxval) / sumval;
  }
}


extern "C" __global__ void softmax_baseline32(float* out, const float* inp, int n, int c) { softmax_forward_baseline_kernel<32>(out,inp,n,c); }

extern "C" __global__ void softmax_online232(float* out, const float* inp, int n, int c) { softmax_forward_online_kernel2<32>(out,inp,n,c); }

extern "C" __global__ void softmax_baseline64(float* out, const float* inp, int n, int c) { softmax_forward_baseline_kernel<64>(out,inp,n,c); }

extern "C" __global__ void softmax_online264(float* out, const float* inp, int n, int c) { softmax_forward_online_kernel2<64>(out,inp,n,c); }

extern "C" __global__ void softmax_online3(float* out,const float* inp,int n,int c) { softmax_forward_online_kernel3(out,inp,n,c); }

// CUDA's 32-lane cooperative SumMax reduction expressed with HIP subwarp shuffles.
extern "C" __global__ void softmax_online(float* out,const float* inp,int n,int c) {
    int lane=threadIdx.x%32, row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(row>=n)return;
    float maxval=-INFINITY, sum=0;
    for(int i=lane;i<c;i+=32) { float v=inp[row*c+i], m=fmaxf(maxval,v); sum=sum*expf(maxval-m)+expf(v-m); maxval=m; }
    for(int d=16;d>0;d/=2) { float v=__shfl_down(maxval,d,32), s=__shfl_down(sum,d,32), m=fmaxf(maxval,v); sum=sum*expf(maxval-m)+s*expf(v-m); maxval=m; }
    maxval=__shfl(maxval,0,32); sum=__shfl(sum,0,32);
    for(int i=lane;i<c;i+=32)out[row*c+i]=expf(inp[row*c+i]-maxval)/sum;
}
