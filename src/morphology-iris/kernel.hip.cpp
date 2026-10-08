#include <hip/hip_runtime.h>
#include <stddef.h>

#define BLACK 0
#define WHITE 255
#define MAX_SEL_SIZE 64

enum class MorphOpType {
  ERODE,
  DILATE,
};

template <MorphOpType opType>
__device__ unsigned char element_op(unsigned char lhs, unsigned char rhs);

template <>
__device__ unsigned char element_op<MorphOpType::ERODE>(unsigned char lhs, unsigned char rhs)
{
  return min(lhs, rhs);
}

template <>
__device__ unsigned char element_op<MorphOpType::DILATE>(unsigned char lhs, unsigned char rhs)
{
  return max(lhs, rhs);
}

template <MorphOpType opType>
__device__ unsigned char border_value();

template <>
__device__ unsigned char border_value<MorphOpType::ERODE>()
{
  return BLACK;
}

template <>
__device__ unsigned char border_value<MorphOpType::DILATE>()
{
  return WHITE;
}

template <MorphOpType opType>
__device__ void two_way_scan(const unsigned char* buffer,
                             unsigned char* op_array,
                             const int sel_size,
                             const int tid)
{
  op_array[tid] = buffer[tid];
  op_array[tid + sel_size] = buffer[tid + sel_size];

  unsigned char fwd_val = 0;
  unsigned char rev_val = 0;
  const int ub = sel_size - 1;

  for (int offset = 1; offset < sel_size; offset *= 2) {
    const int lb = ub - offset;
    const bool is_fwd = tid >= offset;
    const bool is_rev = tid <= lb;

    __syncthreads();

    if (is_fwd) {
      fwd_val = element_op<opType>(op_array[tid + ub], op_array[tid + lb]);
    }
    if (is_rev) {
      rev_val = element_op<opType>(op_array[tid], op_array[tid + offset]);
    }

    __syncthreads();

    if (is_fwd) {
      op_array[tid + ub] = fwd_val;
    }
    if (is_rev) {
      op_array[tid] = rev_val;
    }
  }
}

extern "C" __global__ void clear_u8(unsigned char* dst, size_t n)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = 0;
  }
}

template <MorphOpType opType>
__device__ void vhgw_horiz(unsigned char* __restrict__ dst,
                           const unsigned char* __restrict__ src,
                           const int width,
                           const int height,
                           const int sel_size)
{
  __shared__ unsigned char buffer[2 * MAX_SEL_SIZE];
  __shared__ unsigned char op_array[2 * MAX_SEL_SIZE];

  const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  if (threadIdx.x >= sel_size) {
    return;
  }

  if (tidx < width && tidy < height) {
    buffer[threadIdx.x] = src[tidy * width + tidx];
  } else {
    buffer[threadIdx.x] = 0;
  }
  if (tidx + sel_size < width && tidy < height) {
    buffer[threadIdx.x + sel_size] = src[tidy * width + tidx + sel_size];
  } else {
    buffer[threadIdx.x + sel_size] = 0;
  }
  __syncthreads();

  if (tidx >= width || tidy >= height) {
    return;
  }

  two_way_scan<opType>(buffer, op_array, sel_size, threadIdx.x);

  if (tidx + sel_size / 2 < width - sel_size / 2) {
    dst[tidy * width + tidx + sel_size / 2] =
        element_op<opType>(op_array[threadIdx.x], op_array[threadIdx.x + sel_size - 1]);
  }
}

template <MorphOpType opType>
__device__ void vhgw_vert(unsigned char* __restrict__ dst,
                          const unsigned char* __restrict__ src,
                          const int width,
                          const int height,
                          const int sel_size)
{
  __shared__ unsigned char buffer[2 * MAX_SEL_SIZE];
  __shared__ unsigned char op_array[2 * MAX_SEL_SIZE];

  const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  if (threadIdx.y >= sel_size) {
    return;
  }

  if (tidy < height && tidx < width) {
    buffer[threadIdx.y] = src[tidy * width + tidx];
  } else {
    buffer[threadIdx.y] = 0;
  }
  if (tidy + sel_size < height && tidx < width) {
    buffer[threadIdx.y + sel_size] = src[(tidy + sel_size) * width + tidx];
  } else {
    buffer[threadIdx.y + sel_size] = 0;
  }
  __syncthreads();

  if (tidy >= height || tidx >= width) {
    return;
  }

  two_way_scan<opType>(buffer, op_array, sel_size, threadIdx.y);

  if (tidy + sel_size / 2 < height - sel_size / 2) {
    dst[(tidy + sel_size / 2) * width + tidx] =
        element_op<opType>(op_array[threadIdx.y], op_array[threadIdx.y + sel_size - 1]);
  }

  if (tidy < sel_size / 2 || tidy >= height - sel_size / 2) {
    dst[tidy * width + tidx] = border_value<opType>();
  }
}

extern "C" __global__ void dilate_horiz(unsigned char* dst,
                                         const unsigned char* src,
                                         int width,
                                         int height,
                                         int sel_size)
{
  vhgw_horiz<MorphOpType::DILATE>(dst, src, width, height, sel_size);
}

extern "C" __global__ void dilate_vert(unsigned char* dst,
                                        const unsigned char* src,
                                        int width,
                                        int height,
                                        int sel_size)
{
  vhgw_vert<MorphOpType::DILATE>(dst, src, width, height, sel_size);
}

extern "C" __global__ void erode_horiz(unsigned char* dst,
                                        const unsigned char* src,
                                        int width,
                                        int height,
                                        int sel_size)
{
  vhgw_horiz<MorphOpType::ERODE>(dst, src, width, height, sel_size);
}

extern "C" __global__ void erode_vert(unsigned char* dst,
                                       const unsigned char* src,
                                       int width,
                                       int height,
                                       int sel_size)
{
  vhgw_vert<MorphOpType::ERODE>(dst, src, width, height, sel_size);
}
