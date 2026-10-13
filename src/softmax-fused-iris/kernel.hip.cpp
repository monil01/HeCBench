#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include "reference.h"

#define C10_WARP_SIZE 32
typedef __hip_bfloat16 BFloat16;
typedef __half Half;

template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);

template <>
__device__ __inline__ void copy_vector<BFloat16, 1>(BFloat16 *dst,
                                                    const BFloat16 *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<BFloat16, 4>(BFloat16 *dst,
                                                    const BFloat16 *src) {
  *((float2 *)dst) = *((float2 *)src);
}

template <>
__device__ __inline__ void copy_vector<Half, 1>(Half *dst, const Half *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<Half, 4>(Half *dst, const Half *src) {
  *((float2 *)dst) = *((float2 *)src);
}

template <>
__device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t *dst,
                                                   const uint8_t *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t *dst,
                                                   const uint8_t *src) {
  *((uchar4 *)dst) = *((uchar4 *)src);
}

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value)
    ++log2_value;
  return log2_value;
}

template <typename T> struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T> struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T
WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize) {
  return __shfl_xor(value, laneMask, width);
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE,
          template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t *sum) {
  ReduceOp<acc_t> r;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, WARP_SIZE);
      sum[i] = r(sum[i], b);
    }
  }
}

/*
 * Extended softmax (from native aten pytorch) with following additional
 * features 1) input scaling 2) Explicit masking
 */

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
__device__ void scaled_masked_softmax_warp_forward(
    output_t *dst, const input_t *src, const uint8_t *mask, const acc_t scale,
    int micro_batch_size, int element_count, int pad_batches) {
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
  constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

  // blockDim/threadIdx = (WARP_SIZE, WARPS_PER_BLOCK, 1)
  // gridDim/blockIdx = (seq_len, attn_heads, batches)
  int first_batch = (blockDim.y * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) + threadIdx.y) * WARP_BATCH;

  int pad_first_batch;
  if (pad_batches != 1) { // bert style
    pad_first_batch = (blockDim.y * (blockIdx.x + gridDim.x * blockIdx.z) + threadIdx.y) * WARP_BATCH;
  } else { // gpt2 style
    pad_first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
  }

  // micro_batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = micro_batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  src += first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;
  dst += first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;
  mask += pad_first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;

  // load data from global memory
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  input_t temp_data[ELEMENTS_PER_LDG_STG];
  uint8_t temp_mask[ELEMENTS_PER_LDG_STG];
#pragma unroll
  for (int i = 0; i < local_batches; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {

        int itr_idx = i * element_count + it * WARP_SIZE;
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(temp_data, src + itr_idx);
        copy_vector<uint8_t, ELEMENTS_PER_LDG_STG>(temp_mask, mask + itr_idx);

#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          if (temp_mask[element] != 1) {
            elements[i][it + element] = (acc_t)temp_data[element] * scale;
          } else {
            elements[i][it + element] = (acc_t)-10000.0;
          }
        }
      } else {
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          elements[i][it + element] = -std::numeric_limits<acc_t>::infinity();
        }
      }
    }
  }

  // compute max_value
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

  // compute scale value to account for full mask
  acc_t mask_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    mask_value[i] = (max_value[i] == (acc_t)-10000.0) ? (acc_t)0.0 : (acc_t)1.0;
  }


  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = std::exp((elements[i][it] - max_value[i]));
      sum[i] += elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

  // store result
  output_t out[ELEMENTS_PER_LDG_STG];
#pragma unroll
  for (int i = 0; i < local_batches; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          if (mask_value[i])
            out[element] = elements[i][it + element] / sum[i];
          else
            out[element] = (output_t)0;
        }
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(
            dst + i * element_count + it * WARP_SIZE, out);
      } else {
        break;
      }
    }
  }
}


extern "C" __global__ void fused_half_0(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,0>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_1(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,1>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_2(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,2>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_3(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,3>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_4(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,4>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_5(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,5>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_6(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,6>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_7(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,7>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_8(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,8>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_9(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,9>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_half_10(Half* dst,const Half* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<Half,Half,float,10>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_0(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,0>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_1(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,1>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_2(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,2>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_3(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,3>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_4(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,4>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_5(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,5>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_6(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,6>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_7(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,7>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_8(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,8>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_9(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,9>(dst,src,mask,scale,batches,count,pad); }

extern "C" __global__ void fused_bf_10(BFloat16* dst,const BFloat16* src,const uint8_t* mask,float scale,int batches,int count,int pad) { scaled_masked_softmax_warp_forward<BFloat16,BFloat16,float,10>(dst,src,mask,scale,batches,count,pad); }
