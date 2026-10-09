#include <hip/hip_runtime.h>
#include <stdint.h>

template <typename scalar_t>
__device__ void flip_kernel_impl(
    const scalar_t* in_tensor,
    scalar_t* out_tensor,
    int64_t n,
    const int64_t* flip_dims,
    int64_t flip_dims_size,
    const int64_t* strides,
    const int64_t* strides_contiguous,
    const int64_t* shape,
    int64_t total_dims)
{
  int64_t linear_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear_index >= n) return;

  int64_t cur_indices = linear_index;
  int64_t dst_offset = 0;
  for (int64_t i = 0; i < total_dims; i++) {
    int64_t temp = cur_indices;
    cur_indices = cur_indices / strides_contiguous[i];
    int64_t rem = temp - cur_indices * strides_contiguous[i];
    for (int64_t j = 0; j < flip_dims_size; j++) {
      if (i == flip_dims[j]) {
        cur_indices = shape[i] - 1 - cur_indices;
      }
    }
    dst_offset += cur_indices * strides[i];
    cur_indices = rem;
  }
  out_tensor[linear_index] = in_tensor[dst_offset];
}

extern "C" __global__ void flip_kernel_float(
    const float* in_tensor,
    float* out_tensor,
    int64_t n,
    const int64_t* flip_dims,
    int64_t flip_dims_size,
    const int64_t* strides,
    const int64_t* strides_contiguous,
    const int64_t* shape,
    int64_t total_dims)
{
  flip_kernel_impl(in_tensor, out_tensor, n, flip_dims, flip_dims_size, strides, strides_contiguous, shape, total_dims);
}

extern "C" __global__ void flip_kernel_double(
    const double* in_tensor,
    double* out_tensor,
    int64_t n,
    const int64_t* flip_dims,
    int64_t flip_dims_size,
    const int64_t* strides,
    const int64_t* strides_contiguous,
    const int64_t* shape,
    int64_t total_dims)
{
  flip_kernel_impl(in_tensor, out_tensor, n, flip_dims, flip_dims_size, strides, strides_contiguous, shape, total_dims);
}
