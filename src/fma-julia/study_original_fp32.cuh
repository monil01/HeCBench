// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Exact FP32 specialization excerpts; original kernels.h unchanged.
#include <cuda_runtime.h>
#include <cstdint>
template<typename T> __global__ void implicit_fma_kernel(const T* __restrict__,const T* __restrict__,T* __restrict__,const int* __restrict__,const int* __restrict__,int,int,int,int);
template<typename T> __global__ void implicit_fma_kernel_rowwise(const T* __restrict__,const T* __restrict__,T* __restrict__,const int* __restrict__,const int* __restrict__,int,int,int,int);
template <>
__global__ void implicit_fma_kernel<float>(const float* __restrict__ a,
                                           const float* __restrict__ b,
                                           float* __restrict__ c,
                                           const int* __restrict__ in_indices,
                                           const int* __restrict__ out_indices,
                                           int num_ops,
                                           int C,
                                           int N_A,
                                           int N_C) {
  // Shared memory for vector B
  extern __shared__ char shared_mem_raw[];
  float* shared_b = reinterpret_cast<float*>(shared_mem_raw);

  // Load vector B into shared memory cooperatively using float4
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;
  int C_vec4 = C / 4;
  int C_remainder = C % 4;

  // Load 4 elements at a time with alignment checking
  for (int i = tid; i < C_vec4; i += threads_per_block) {
    const float* b_ptr = &b[i * 4];
    float* sb_ptr = &shared_b[i * 4];

    if (reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(sb_ptr) % 16 == 0) {
      // Aligned - use vectorized load
      float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
      *reinterpret_cast<float4*>(sb_ptr) = b_vec;
    } else {
      // Not aligned - use scalar loads
      for (int j = 0; j < 4; ++j) {
        shared_b[i * 4 + j] = b[i * 4 + j];
      }
    }
  }

  // Load remaining elements
  int start_remainder = C_vec4 * 4;
  for (int i = tid; i < C_remainder; i += threads_per_block) {
    shared_b[start_remainder + i] = b[start_remainder + i];
  }
  __syncthreads();

  // Modified thread mapping: each thread processes 4 consecutive channels
  int total_vec4_threads = num_ops * C_vec4;
  int total_remainder_threads = num_ops * (C_remainder > 0 ? 1 : 0);
  int total_threads = total_vec4_threads + total_remainder_threads;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_threads) return;

  if (idx < total_vec4_threads) {
    // Vectorized path: process 4 consecutive channels
    int op_idx = idx / C_vec4;
    int vec_idx = idx % C_vec4;

    // Get the indices for this operation
    int in_idx = in_indices[op_idx];
    int out_idx = out_indices[op_idx];

    // Bounds checking
    if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
      return;
    }

    // Check alignment for float4 operations
    const float* a_ptr = &a[in_idx * C + vec_idx * 4];
    float* c_ptr = &c[out_idx * C + vec_idx * 4];
    const float* b_ptr = &shared_b[vec_idx * 4];

    if (reinterpret_cast<uintptr_t>(a_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(c_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0) {
      // Aligned - use vectorized operations
      float4 a_vec = *reinterpret_cast<const float4*>(a_ptr);
      float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
      float4 c_vec = *reinterpret_cast<const float4*>(c_ptr);

      // Perform vectorized FMA: c += a * b
      c_vec.x += a_vec.x * b_vec.x;
      c_vec.y += a_vec.y * b_vec.y;
      c_vec.z += a_vec.z * b_vec.z;
      c_vec.w += a_vec.w * b_vec.w;

      // Store result back
      *reinterpret_cast<float4*>(c_ptr) = c_vec;
    } else {
      // Not aligned - fall back to scalar operations
      for (int i = 0; i < 4; ++i) {
        int ch_idx = vec_idx * 4 + i;
        float a_val = a[in_idx * C + ch_idx];
        float b_val = shared_b[ch_idx];
        c[out_idx * C + ch_idx] += a_val * b_val;
      }
    }
  } else if (C_remainder > 0) {
    // Scalar path: process remaining channels
    int remainder_idx = idx - total_vec4_threads;
    int op_idx = remainder_idx;  // One thread per operation for remainder

    // Get the indices for this operation
    int in_idx = in_indices[op_idx];
    int out_idx = out_indices[op_idx];

    // Bounds checking
    if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
      return;
    }

    // Process remaining channels
    int start_ch = C_vec4 * 4;
    for (int ch_offset = 0; ch_offset < C_remainder; ++ch_offset) {
      int ch_idx = start_ch + ch_offset;
      float a_val = a[in_idx * C + ch_idx];
      float b_val = shared_b[ch_idx];
      c[out_idx * C + ch_idx] += a_val * b_val;
    }
  }
}

template <>
__global__ void implicit_fma_kernel_rowwise<float>(const float* __restrict__ a,
                                                   const float* __restrict__ b,
                                                   float* __restrict__ c,
                                                   const int* __restrict__ in_indices,
                                                   const int* __restrict__ out_indices,
                                                   int num_ops,
                                                   int C,
                                                   int N_A,
                                                   int N_C) {
  // Shared memory for vector B
  extern __shared__ char shared_mem_raw[];
  float* shared_b = reinterpret_cast<float*>(shared_mem_raw);

  // Load vector B into shared memory cooperatively using float4
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;
  int C_vec4 = C / 4;
  int C_remainder = C % 4;

  // Load 4 elements at a time with alignment checking
  for (int i = tid; i < C_vec4; i += threads_per_block) {
    const float* b_ptr = &b[i * 4];
    float* sb_ptr = &shared_b[i * 4];

    if (reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(sb_ptr) % 16 == 0) {
      // Aligned - use vectorized load
      float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
      *reinterpret_cast<float4*>(sb_ptr) = b_vec;
    } else {
      // Not aligned - use scalar loads
      for (int j = 0; j < 4; ++j) {
        shared_b[i * 4 + j] = b[i * 4 + j];
      }
    }
  }

  // Load remaining elements
  int start_remainder = C_vec4 * 4;
  for (int i = tid; i < C_remainder; i += threads_per_block) {
    shared_b[start_remainder + i] = b[start_remainder + i];
  }
  __syncthreads();

  // Each thread processes one operation (one row)
  int op_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (op_idx >= num_ops) return;

  // Get the indices for this operation
  int in_idx = in_indices[op_idx];
  int out_idx = out_indices[op_idx];

  // Bounds checking
  if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
    return;
  }

  // Process channels using float4 vectorization
  int ch_idx = 0;

  // Process 4 channels at a time
  for (int vec_idx = 0; vec_idx < C_vec4; ++vec_idx) {
    // Check alignment for float4 operations
    const float* a_ptr = &a[in_idx * C + vec_idx * 4];
    float* c_ptr = &c[out_idx * C + vec_idx * 4];
    const float* b_ptr = &shared_b[vec_idx * 4];

    if (reinterpret_cast<uintptr_t>(a_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(c_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0) {
      // Aligned - use vectorized operations
      float4 a_vec = *reinterpret_cast<const float4*>(a_ptr);
      float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
      float4 c_vec = *reinterpret_cast<const float4*>(c_ptr);

      // Perform vectorized FMA: c += a * b
      c_vec.x += a_vec.x * b_vec.x;
      c_vec.y += a_vec.y * b_vec.y;
      c_vec.z += a_vec.z * b_vec.z;
      c_vec.w += a_vec.w * b_vec.w;

      // Store result back
      *reinterpret_cast<float4*>(c_ptr) = c_vec;
    } else {
      // Not aligned - fall back to scalar operations
      for (int i = 0; i < 4; ++i) {
        int ch_idx = vec_idx * 4 + i;
        float a_val = a[in_idx * C + ch_idx];
        float b_val = shared_b[ch_idx];
        c[out_idx * C + ch_idx] += a_val * b_val;
      }
    }
  }

  // Process remaining channels
  int start_ch = C_vec4 * 4;
  for (int ch_offset = 0; ch_offset < C_remainder; ++ch_offset) {
    ch_idx = start_ch + ch_offset;
    float a_val = a[in_idx * C + ch_idx];
    float b_val = shared_b[ch_idx];
    c[out_idx * C + ch_idx] += a_val * b_val;
  }
}
