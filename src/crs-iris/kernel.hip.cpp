#include <hip/hip_runtime.h>
#include "utils.h"
static_assert(sizeof(long)==8);
extern "C" __global__ void gcrs_m_1_w_4_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 4;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    __syncthreads();

  }

  out[idx] = result;
}

extern "C" __global__ void gcrs_m_1_w_5_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 5;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    __syncthreads();

  }

  out[idx] = result;
}

extern "C" __global__ void gcrs_m_1_w_6_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 6;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    __syncthreads();

  }

  out[idx] = result;
}

extern "C" __global__ void gcrs_m_1_w_7_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 7;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    __syncthreads();

  }

  out[idx] = result;
}

extern "C" __global__ void gcrs_m_1_w_8_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 8;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    __syncthreads();

  }

  out[idx] = result;
}

extern "C" __global__ void gcrs_m_2_w_4_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 4;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

extern "C" __global__ void gcrs_m_2_w_5_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 5;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

extern "C" __global__ void gcrs_m_2_w_6_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 6;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

extern "C" __global__ void gcrs_m_2_w_7_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 7;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

extern "C" __global__ void gcrs_m_2_w_8_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 8;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

extern "C" __global__ void gcrs_m_3_w_4_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 4;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

extern "C" __global__ void gcrs_m_3_w_5_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 5;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

extern "C" __global__ void gcrs_m_3_w_6_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 6;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

extern "C" __global__ void gcrs_m_3_w_7_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 7;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

extern "C" __global__ void gcrs_m_3_w_8_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 8;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

extern "C" __global__ void gcrs_m_4_w_4_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 4;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

extern "C" __global__ void gcrs_m_4_w_5_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 5;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

extern "C" __global__ void gcrs_m_4_w_6_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 6;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

extern "C" __global__ void gcrs_m_4_w_7_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 7;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

extern "C" __global__ void gcrs_m_4_w_8_coding_dotprod(
  int k, int index,
  const long *__restrict in,
  long *__restrict out,
  const unsigned int *__restrict bm,
  int size)
{
  __shared__ long shared_data[MAX_THREAD_NUM];

  int w = 8;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = blockDim.x / w * w;
  const unsigned int idx = worksize_perblock * blockIdx.x + threadIdx.x;

  if (threadIdx.x >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (threadIdx.x / w) * w;
  int group_inner_offset = threadIdx.x % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[threadIdx.x] = *(in + i*size + idx);

    __syncthreads();

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    __syncthreads();

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

