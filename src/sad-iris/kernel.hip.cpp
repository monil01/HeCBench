#include <hip/hip_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE (BLOCK_SIZE_X * BLOCK_SIZE_Y)
#define FOUND_MIN 5000
#define min_value(a, b) ((a) < (b) ? (a) : (b))

extern "C" __global__ void compute_sad_array(
    int* __restrict__ sad_array,
    const unsigned char* __restrict__ image,
    const unsigned char* __restrict__ kernel,
    const int sad_array_size,
    const int image_width,
    const int image_height,
    const int kernel_width,
    const int kernel_height,
    const int kernel_size)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int sad_result = 0;

  if (row < image_height && col < image_width) {
    const int overlap_width = min_value(image_width - col, kernel_width);
    const int overlap_height = min_value(image_height - row, kernel_height);
    #pragma unroll 4
    for (int kr = 0; kr < overlap_height; kr++) {
      #pragma unroll 4
      for (int kc = 0; kc < overlap_width; kc++) {
        const int image_addr = ((row + kr) * image_width + (col + kc)) * 3;
        const int kernel_addr = (kr * kernel_width + kc) * 3;
        const int m_r = static_cast<int>(image[image_addr + 0]);
        const int m_g = static_cast<int>(image[image_addr + 1]);
        const int m_b = static_cast<int>(image[image_addr + 2]);
        const int t_r = static_cast<int>(kernel[kernel_addr + 0]);
        const int t_g = static_cast<int>(kernel[kernel_addr + 1]);
        const int t_b = static_cast<int>(kernel[kernel_addr + 2]);
        const int error = abs(m_r - t_r) + abs(m_g - t_g) + abs(m_b - t_b);
        sad_result += error;
      }
    }

    int norm_sad = static_cast<int>(sad_result / static_cast<float>(kernel_size));
    int sad_index = row * image_width + col;
    if (sad_index < sad_array_size) {
      sad_array[sad_index] = norm_sad;
    }
  }
}

extern "C" __global__ void find_min_in_sad_array(
    const int sad_array_size,
    const int* __restrict__ sad_array,
    int* __restrict__ min_sad)
{
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  __shared__ int cache[BLOCK_SIZE];
  int temp = FOUND_MIN;
  while (gid + offset < static_cast<unsigned int>(sad_array_size)) {
    temp = min_value(temp, sad_array[gid + offset]);
    offset += stride;
  }

  cache[threadIdx.x] = temp;
  __syncthreads();

  unsigned int i = blockDim.x / 2;
  while (i != 0) {
    if (threadIdx.x < i) {
      cache[threadIdx.x] = min_value(cache[threadIdx.x], cache[threadIdx.x + i]);
    }
    __syncthreads();
    i /= 2;
  }

  if (threadIdx.x == 0) {
    atomicMin(min_sad, cache[0]);
  }
}

extern "C" __global__ void get_num_of_occurrences(
    const int sad_array_size,
    const int* __restrict__ sad_array,
    const int* __restrict__ min_sad,
    int* __restrict__ num_occurrences)
{
  unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int s;

  if (gid < static_cast<unsigned int>(sad_array_size)) {
    if (threadIdx.x == 0) {
      s = 0;
    }
    __syncthreads();

    if (sad_array[gid] == *min_sad) {
      atomicAdd(&s, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      atomicAdd(num_occurrences, s);
    }
  }
}
