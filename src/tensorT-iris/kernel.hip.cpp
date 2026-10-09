#include <hip/hip_runtime.h>

#define TILE_SIZE 5900

extern "C" __global__ void tensor_transpose(
    const int dim_input,
    const int dim_output,
    const int nblocks,
    const int tile_size,
    const int* shape_input,
    const int* shape_output,
    const float* shape_input_r,
    const float* shape_output_r,
    const int* stride_input,
    const int* stride_output_local,
    const int* stride_output_global,
    const double* input,
    double* output)
{
  __shared__ double tile[TILE_SIZE];

  for (int block_idx = blockIdx.x; block_idx < nblocks; block_idx += gridDim.x) {
    int it = block_idx, im = 0, offset1 = 0;
    for (int i = 0; i < dim_input; i++) {
      im = it * shape_input_r[i];
      offset1 += stride_input[i] * (it - im * shape_input[i]);
      it = im;
    }

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
      tile[i] = input[i + block_idx * tile_size];
    }

    __syncthreads();

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
      it = i;
      int offset2 = 0, local_offset = 0;
      for (int j = 0; j < dim_output; j++) {
        im = it * shape_output_r[j];
        int tmp = it - im * shape_output[j];
        offset2 += stride_output_global[j] * tmp;
        local_offset += stride_output_local[j] * tmp;
        it = im;
      }
      output[offset1 + offset2] = tile[local_offset];
    }

    __syncthreads();
  }
}
