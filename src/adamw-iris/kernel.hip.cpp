#include <hip/hip_runtime.h>

#include <stdint.h>

static __device__ __constant__ uint8_t bitmask = 15;
static __device__ __constant__ uint8_t right_pack_bitmask = 15 << 4;

static __device__ __constant__ float exp_qmap[16] = {
    -0.8875f, -0.6625f, -0.4375f, -0.2125f, -0.0775f, -0.0325f, -0.0055f, 0.0000f,
    0.0055f,  0.0325f,  0.0775f,  0.2125f,  0.4375f,  0.6625f,  0.8875f, 1.0000f,
};

static __device__ __constant__ float exp_qmidpt[15] = {
    -0.775f, -0.55f, -0.325f, -0.145f, -0.055f, -0.019f, -0.00275f, 0.00275f,
    0.019f,  0.055f, 0.145f,  0.325f,  0.55f,   0.775f,  0.94375f,
};

static __device__ __constant__ float sq_qmap[16] = {
    0.0625f, 0.1250f, 0.1875f, 0.2500f, 0.3125f, 0.3750f, 0.4375f, 0.5000f,
    0.5625f, 0.6250f, 0.6875f, 0.7500f, 0.8125f, 0.8750f, 0.9375f, 1.0000f,
};

static __device__ __constant__ float sq_qmidpt[15] = {
    0.09375f, 0.15625f, 0.21875f, 0.28125f, 0.34375f, 0.40625f, 0.46875f, 0.53125f,
    0.59375f, 0.65625f, 0.71875f, 0.78125f, 0.84375f, 0.90625f, 0.96875f,
};

static __device__ __forceinline__ int q_mapping(const float* qmap, const float* qmidpt, float x)
{
  int low = 0;
  int high = 15;

  if (x <= qmap[low]) return low;
  if (qmap[high] <= x) return high;

  while (low < high) {
    int mid = (low + high) >> 1;
    if (qmap[mid] <= x)
      low = mid + 1;
    else
      high = mid;
  }
  return (qmidpt[low - 1] < x) ? low : low - 1;
}

extern "C" __global__ void fused_4bit_kernel(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ exp_qscale,
    float* __restrict__ sq_qscale,
    int8_t* __restrict__ exp,
    int8_t* __restrict__ sq,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float step,
    const long total_size,
    const float correction1,
    const float correction2_sqrt,
    const float step_size,
    const float weight_decay_update,
    const float resid_beta1,
    const float resid_beta2)
{
  (void)lr;
  (void)weight_decay;
  (void)step;
  (void)correction1;

  __shared__ float exp_left_cache[64];
  __shared__ float exp_right_cache[64];
  __shared__ float sq_left_cache[64];
  __shared__ float sq_right_cache[64];
  __shared__ float reduce_exp[64];
  __shared__ float reduce_sq[64];
  __shared__ float absmax_exp;
  __shared__ float absmax_sq;

  const int tid = threadIdx.x;
  const long global_id = static_cast<long>(blockIdx.x) * blockDim.x + tid;
  const bool active = global_id < total_size;

  float local_absmax_exp = 0.0f;
  float local_absmax_sq = 0.0f;
  float local_exp_left = 0.0f;
  float local_exp_right = 0.0f;
  float local_sq_left = 0.0f;
  float local_sq_right = 0.0f;

  if (active) {
    const int8_t exp_full = exp[global_id];
    const int8_t sq_full = sq[global_id];

    const float p_left = p[global_id * 2];
    const float p_right = p[global_id * 2 + 1];
    const float g_left = g[global_id * 2];
    const float g_right = g[global_id * 2 + 1];

    const int8_t exp_left_index = exp_full & bitmask;
    const int8_t sq_left_index = sq_full & bitmask;

    const float p_left_decay = p_left * weight_decay_update;
    const float exp_avg_qscale = exp_qscale[blockIdx.x];

    local_exp_left = exp_qmap[exp_left_index] * exp_avg_qscale;
    local_exp_left = beta1 * local_exp_left + resid_beta1 * g_left;

    local_sq_left = sq_qmap[sq_left_index] * sq_qscale[blockIdx.x];
    local_sq_left = beta2 * local_sq_left + resid_beta2 * (g_left * g_left);

    p[global_id * 2] =
        p_left_decay - (step_size * (local_exp_left / (sqrtf(local_sq_left) / correction2_sqrt + eps)));

    const int8_t exp_right_index = (exp_full >> 4) & bitmask;
    const int8_t sq_right_index = (sq_full >> 4) & bitmask;

    const float p_right_decay = p_right * weight_decay_update;

    local_exp_right = exp_qmap[exp_right_index] * exp_avg_qscale;
    local_exp_right = beta1 * local_exp_right + resid_beta1 * g_right;

    local_sq_right = sq_qmap[sq_right_index] * sq_qscale[blockIdx.x];
    local_sq_right = beta2 * local_sq_right + resid_beta2 * (g_right * g_right);

    p[global_id * 2 + 1] =
        p_right_decay - (step_size * (local_exp_right / (sqrtf(local_sq_right) / correction2_sqrt + eps)));

    local_absmax_exp = fmaxf(local_exp_left, local_exp_right);
    local_absmax_sq = fmaxf(local_sq_left, local_sq_right);
  }

  exp_left_cache[tid] = local_exp_left;
  exp_right_cache[tid] = local_exp_right;
  sq_left_cache[tid] = local_sq_left;
  sq_right_cache[tid] = local_sq_right;
  reduce_exp[tid] = local_absmax_exp;
  reduce_sq[tid] = local_absmax_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_exp[tid] = fmaxf(reduce_exp[tid], reduce_exp[tid + stride]);
      reduce_sq[tid] = fmaxf(reduce_sq[tid], reduce_sq[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    exp_qscale[blockIdx.x] = reduce_exp[0];
    sq_qscale[blockIdx.x] = reduce_sq[0];
    absmax_exp = reduce_exp[0];
    absmax_sq = reduce_sq[0];
  }
  __syncthreads();

  if (active) {
    int8_t local_packed_exp = 0;
    int8_t local_packed_sq = 0;

    const int8_t q_exp_left =
        static_cast<int8_t>(q_mapping(exp_qmap, exp_qmidpt, exp_left_cache[tid] / absmax_exp));
    const int8_t q_sq_left =
        static_cast<int8_t>(q_mapping(sq_qmap, sq_qmidpt, sq_left_cache[tid] / absmax_sq));
    local_packed_exp |= (q_exp_left & bitmask);
    local_packed_sq |= (q_sq_left & bitmask);

    const int8_t q_exp_right =
        static_cast<int8_t>(q_mapping(exp_qmap, exp_qmidpt, exp_right_cache[tid] / absmax_exp));
    const int8_t q_sq_right =
        static_cast<int8_t>(q_mapping(sq_qmap, sq_qmidpt, sq_right_cache[tid] / absmax_sq));
    local_packed_exp |= (q_exp_right & right_pack_bitmask);
    local_packed_sq |= (q_sq_right & right_pack_bitmask);

    exp[global_id] = local_packed_exp;
    sq[global_id] = local_packed_sq;
  }
}
