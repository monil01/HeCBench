#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "node.h"

__device__
float LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

__device__
void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

extern "C" __global__ void init_random_full(unsigned int seed, unsigned int *state)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  state[i] = seed ^ i;
  LCG_random_init(state+i);
}

extern "C" __global__ void random_init(
    unsigned int *__restrict__ state,
    int state_size,
    half *__restrict__ array,
    long long array_size,
    long long k,
    float scale)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int state_id = tid % state_size;
  for(int i = 0;i < array_size;i += gridDim.x*blockDim.x)
  {
    int idx = i + tid;
    if(idx >= array_size) break;
    array[idx] = __float2half(LCG_random(state+state_id)*scale);
  }
}

extern "C" __global__ void init_random_bounded(unsigned int seed, unsigned int *state, int size)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < size) {
    state[i] = seed ^ i;
    LCG_random_init(state+i);
  }
}

extern "C" __global__ void transform_half(
  const half *__restrict__ gpu_half_feature,
  float *__restrict__ gpu_float_feature,
  long long vec_size)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int number_threads = gridDim.x*blockDim.x;

  for(long long i = tid;i < vec_size;i += number_threads)
  {
    gpu_float_feature[i] = __half2float(gpu_half_feature[i]);
  }
}

#include "hip/hip_runtime.h"
extern "C" __global__ void sgd_k128_kernel_hogwild_warp32_lrate(
    const mf_node *__restrict__ R,
    long long nnz,
    half *__restrict__ p,
    half *__restrict__ q,
    unsigned int *state,
    const float *__restrict__ dynamic_rate,
    long long u_seg,
    long long v_seg,
    int k,
    int num_iters,
    int current_iter,
    int update_count_per_block,
    int update_count_this_block,
    int update_vector_size,
    float lambda_p,
    float lambda_q,
    int u_grid,
    int v_grid,
    int u_id,
    int v_id)
{
#ifdef HECBENCH_MF_VERIFY_SERIAL_WARPS
  // Validation-only ordering: all four workers and updates remain present.
  for(int validation_warp=0; validation_warp<4; ++validation_warp) {
    if(threadIdx.x/32==validation_warp) {
#endif

  //persistant thread
  for(int ite = current_iter; ite < current_iter + num_iters; ite ++)
  {
    float tmp_lrate = __ldg(&dynamic_rate[ite]);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {

      int lane_id = threadIdx.x%32;
      int local_wid = threadIdx.x/32;
      int wid = 4*blockIdx.x + local_wid;

      long long start_id = 0;
      if(lane_id == 0)
      {
        long long origin = (long long)(LCG_random(state+wid)*nnz);
        start_id = origin%nnz;
      }
      start_id = __shfl(start_id, 0, 32);

      for(int i = 0;i < update_vector_size;i++)
      {
        int offset = (start_id + i)%nnz;

        float r = __ldg(&R[offset].rate);
        int u = __ldg(&R[offset].u);
        int v = __ldg(&R[offset].v);

        //read the p & q into register file.
        int base_p = u*k;
        int base_q = v*k;

        float tmp_p1 = __half2float(p[base_p + lane_id]);
        float tmp_q1 = __half2float(q[base_q + lane_id]);

        float tmp_p2 = __half2float(p[base_p + lane_id + 32]);
        float tmp_q2 = __half2float(q[base_q + lane_id + 32]);

        float tmp_p3 = __half2float(p[base_p + lane_id + 64]);
        float tmp_q3 = __half2float(q[base_q + lane_id + 64]);

        float tmp_p4 = __half2float(p[base_p + lane_id + 96]);
        float tmp_q4 = __half2float(q[base_q + lane_id + 96]);

        float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;

        //get dot product.
        tmp_product += __shfl_down(tmp_product, 16, 32);
        tmp_product += __shfl_down(tmp_product, 8, 32);
        tmp_product += __shfl_down(tmp_product, 4, 32);
        tmp_product += __shfl_down(tmp_product, 2, 32);
        tmp_product += __shfl_down(tmp_product, 1, 32);
        tmp_product = __shfl(tmp_product,0,32);

        float ruv = r - tmp_product;

        //update
        //only works for k=blockDim.x=128
        p[base_p + lane_id +  0] = __float2half(tmp_p1 + tmp_lrate*(ruv*tmp_q1 - lambda_p*tmp_p1));
        q[base_q + lane_id +  0] = __float2half(tmp_q1 + tmp_lrate*(ruv*tmp_p1 - lambda_q*tmp_q1));

        p[base_p + lane_id + 32] = __float2half(tmp_p2 + tmp_lrate*(ruv*tmp_q2 - lambda_p*tmp_p2));
        q[base_q + lane_id + 32] = __float2half(tmp_q2 + tmp_lrate*(ruv*tmp_p2 - lambda_q*tmp_q2));

        p[base_p + lane_id + 64] = __float2half(tmp_p3 + tmp_lrate*(ruv*tmp_q3 - lambda_p*tmp_p3));
        q[base_q + lane_id + 64] = __float2half(tmp_q3 + tmp_lrate*(ruv*tmp_p3 - lambda_q*tmp_q3));

        p[base_p + lane_id + 96] = __float2half(tmp_p4 + tmp_lrate*(ruv*tmp_q4 - lambda_p*tmp_p4));
        q[base_q + lane_id + 96] = __float2half(tmp_q4 + tmp_lrate*(ruv*tmp_p4 - lambda_q*tmp_q4));
      }
    }
  }

#ifdef HECBENCH_MF_VERIFY_SERIAL_WARPS
    }
    __syncthreads();
  }
#endif
}
