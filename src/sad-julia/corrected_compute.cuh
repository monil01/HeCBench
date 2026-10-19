#pragma once
#define SAD_MIN(a,b) ((a)<(b)?(a):(b))
__global__ void compute_sad_array_corrected(
                    int*__restrict__ sad_array,
    const unsigned char*__restrict__ image,
    const unsigned char*__restrict__ kernel,
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

  if (row < image_height - kernel_height + 1 && col < image_width - kernel_width + 1) {
    const int overlap_width = SAD_MIN(image_width - col, kernel_width);
    const int overlap_height = SAD_MIN(image_height - row, kernel_height);
    #pragma unroll 4
    for (int kr = 0; kr < overlap_height; kr++) {
      #pragma unroll 4
      for (int kc = 0; kc < overlap_width; kc++) {
        const int image_addr = ((row + kr) * image_width + (col + kc)) * 3;
        const int kernel_addr = (kr * kernel_width + kc) * 3;
        const int m_r = (int)(image[image_addr + 0]);
        const int m_g = (int)(image[image_addr + 1]);
        const int m_b = (int)(image[image_addr + 2]);
        const int t_r = (int)(kernel[kernel_addr + 0]);
        const int t_g = (int)(kernel[kernel_addr + 1]);
        const int t_b = (int)(kernel[kernel_addr + 2]);
        const int error = abs(m_r - t_r) + abs(m_g - t_g) + abs(m_b - t_b);
        sad_result += error;
      }
    }

    int norm_sad = (int)(sad_result / (float)kernel_size);

    int my_index_in_sad_array = row * (image_width - kernel_width + 1) + col;
    if (my_index_in_sad_array < sad_array_size) {
      sad_array[my_index_in_sad_array] = norm_sad;
    }
  }
}


#undef SAD_MIN
