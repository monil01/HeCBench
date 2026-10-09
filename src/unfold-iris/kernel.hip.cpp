#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void unfold_backward_kernel(
    int* __restrict__ grad_out_ptr,
    const int* __restrict__ grad_in_ptr,
    const int64_t* __restrict__ idx_dim_ptr,
    int total_n_elems,
    int64_t size,
    int64_t step,
    int64_t grad_in_dim_stride,
    int64_t grad_in_last_dim_stride,
    int64_t grad_in_dim_size,
    int64_t grad_out_dim_stride)
{
  (void)grad_out_dim_stride;

  constexpr int n_elems_per_thread = 4;
  const int base = static_cast<int>((blockIdx.x * blockDim.x) * n_elems_per_thread + threadIdx.x);

  #pragma unroll
  for (int lane = 0; lane < n_elems_per_thread; ++lane) {
    const int i = base + lane * static_cast<int>(blockDim.x);
    if (i >= total_n_elems) continue;

    int* __restrict__ grad_out_data = grad_out_ptr + i;
    const int* __restrict__ grad_in_data = grad_in_ptr + i;
    const int64_t idx_dim = *idx_dim_ptr;

    int64_t left_fold_idx = (idx_dim > size) ? (idx_dim - size) / step : 0;
    if (!(left_fold_idx * step <= idx_dim && idx_dim < left_fold_idx * step + size)) {
      ++left_fold_idx;
    }

    int64_t right_fold_idx = idx_dim / step;
    right_fold_idx = (right_fold_idx >= grad_in_dim_size) ? (grad_in_dim_size - 1) : right_fold_idx;

    for (int64_t fold_idx = left_fold_idx; fold_idx <= right_fold_idx; ++fold_idx) {
      const int64_t idx_last_dim = idx_dim - fold_idx * step;
      *grad_out_data += grad_in_data[fold_idx * grad_in_dim_stride +
                                     idx_last_dim * grad_in_last_dim_stride];
    }
  }
}
