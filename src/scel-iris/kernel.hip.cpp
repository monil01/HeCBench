#include <hip/hip_runtime.h>
#include <math.h>

#define GPU_NUM_THREADS 256

__device__ static float sigmoid_xent_forward(float lgt, float tgt)
{
  return lgt * (tgt - (lgt >= 0.f)) -
         logf(1.f + expf(lgt - 2.f * lgt * (lgt >= 0.f)));
}

__device__ static float sigmoid_partition(float lgt)
{
  return lgt * (lgt >= 0.f) + logf(1.f + expf(lgt - 2.f * lgt * (lgt >= 0.f)));
}

__device__ static float sigmoid_xent_forward_with_log_d_trick(float lgt, float tgt)
{
  return (2.f * tgt - 1.f) * (lgt - sigmoid_partition(lgt));
}

__device__ static float unjoined_sigmoid_xent_forward(float lgt, float tgt)
{
  return lgt * tgt + (tgt - 1.f) * lgt * (lgt >= 0.f) -
         (1.f - tgt) * logf(1.f + expf(lgt - 2.f * lgt * (lgt >= 0.f)));
}

extern "C" __global__ void SigmoidCrossEntropyWithLogitsKernel(
    const int inner_size, const int log_D_trick, const int unjoined_lr_loss,
    const float* logits_ptr, const float* targets_ptr, float* out_ptr)
{
  __shared__ float partial[GPU_NUM_THREADS];
  int i = blockIdx.x;
  int last_idx = (i + 1) * inner_size;
  float value = 0;
  for (int in_idx = i * inner_size + threadIdx.x;
       in_idx < last_idx; in_idx += blockDim.x) {
    float lgt = logits_ptr[in_idx];
    float tgt = targets_ptr[in_idx];
    if (unjoined_lr_loss) {
      value += unjoined_sigmoid_xent_forward(lgt, tgt);
    } else {
      value += log_D_trick ? sigmoid_xent_forward_with_log_d_trick(lgt, tgt)
                           : sigmoid_xent_forward(lgt, tgt);
    }
  }

  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out_ptr[i] = -partial[0] / inner_size;
  }
}
