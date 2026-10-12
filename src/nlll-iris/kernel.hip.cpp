#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

template <typename scalar_t, typename accscalar_t, typename index_t, int NLL_LOSS_THREADS>
__device__ void nll_loss_forward_reduce2d_impl(scalar_t* __restrict__ output,
                                               scalar_t* __restrict__ total_weight,
                                               const scalar_t* __restrict__ input,
                                               const index_t* __restrict__ target,
                                               const scalar_t* __restrict__ weights,
                                               bool size_average,
                                               int64_t nframe,
                                               int64_t kdim,
                                               int64_t ignore_index)
{
  __shared__ accscalar_t sm_inputs[NLL_LOSS_THREADS];
  __shared__ accscalar_t acc_weight[NLL_LOSS_THREADS];

  const int tid = threadIdx.x;
  sm_inputs[tid] = static_cast<accscalar_t>(0);
  acc_weight[tid] = static_cast<accscalar_t>(0);

  for (int64_t i = tid; i < nframe; i += NLL_LOSS_THREADS) {
    const index_t t = target[i];
    if (t != ignore_index) {
      const scalar_t cur_weight = weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
      sm_inputs[tid] -= static_cast<accscalar_t>(input[i * kdim + t] * cur_weight);
      acc_weight[tid] += static_cast<accscalar_t>(cur_weight);
    }
  }

  __syncthreads();

  if (tid == 0) {
    accscalar_t output_acc = 0;
    accscalar_t total_weight_acc = 0;
    for (int i = 0; i < NLL_LOSS_THREADS; ++i) {
      output_acc += sm_inputs[i];
      total_weight_acc += acc_weight[i];
    }
    *total_weight = static_cast<scalar_t>(total_weight_acc);
    *output = size_average ? static_cast<scalar_t>(output_acc / total_weight_acc)
                           : static_cast<scalar_t>(output_acc);
  }
}

#define DEFINE_NLL_FLOAT_KERNEL(THREADS) \
extern "C" __global__ void nll_loss_float_##THREADS( \
    float* output, float* total_weight, const float* input, const int* target, \
    const float* weights, bool size_average, int64_t nframe, int64_t kdim, \
    int64_t ignore_index) \
{ \
  nll_loss_forward_reduce2d_impl<float, float, int, THREADS>( \
      output, total_weight, input, target, weights, size_average, nframe, kdim, \
      ignore_index); \
}

#define DEFINE_NLL_HALF_KERNEL(THREADS) \
extern "C" __global__ void nll_loss_half_##THREADS( \
    __half* output, __half* total_weight, const __half* input, const int* target, \
    const __half* weights, bool size_average, int64_t nframe, int64_t kdim, \
    int64_t ignore_index) \
{ \
  nll_loss_forward_reduce2d_impl<__half, __half, int, THREADS>( \
      output, total_weight, input, target, weights, size_average, nframe, kdim, \
      ignore_index); \
}

DEFINE_NLL_FLOAT_KERNEL(64)
DEFINE_NLL_FLOAT_KERNEL(128)
DEFINE_NLL_FLOAT_KERNEL(256)
DEFINE_NLL_FLOAT_KERNEL(512)
DEFINE_NLL_FLOAT_KERNEL(1024)

DEFINE_NLL_HALF_KERNEL(64)
DEFINE_NLL_HALF_KERNEL(128)
DEFINE_NLL_HALF_KERNEL(256)
DEFINE_NLL_HALF_KERNEL(512)
DEFINE_NLL_HALF_KERNEL(1024)
