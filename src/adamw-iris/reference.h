#include <cmath>
#include <cstdint>

static const uint8_t bitmask = 15;
static const uint8_t right_pack_bitmask = bitmask << 4;

static const float exp_qmap[] = {
    -0.8875, -0.6625, -0.4375, -0.2125, -0.0775, -0.0325, -0.0055, 0.0000,
    0.0055,  0.0325,  0.0775,  0.2125,  0.4375,  0.6625,  0.8875, 1.0000,
};

static const float exp_qmidpt[] = {
    -0.775, -0.55, -0.325, -0.145, -0.055, -0.019, -0.00275, 0.00275,
    0.019,  0.055, 0.145,  0.325,  0.55,   0.775,  0.94375,
};

static const float sq_qmap[] = {
    0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750, 0.4375, 0.5000,
    0.5625, 0.6250, 0.6875, 0.7500, 0.8125, 0.8750, 0.9375, 1.0000,
};

static const float sq_qmidpt[] = {
    0.09375, 0.15625, 0.21875, 0.28125, 0.34375, 0.40625, 0.46875, 0.53125,
    0.59375, 0.65625, 0.71875, 0.78125, 0.84375, 0.90625, 0.96875,
};

static inline int q_mapping(const float* qmap, const float* qmidpt, float x)
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

template <int block_size>
void reference(
    const int grid_size,
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
    const int64_t total_size,
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

  for (int block_id = 0; block_id < grid_size; block_id++) {
    float absmax_exp = 0;
    float absmax_sq = 0;
    float local_exp_left[block_size];
    float local_sq_left[block_size];
    float local_exp_right[block_size];
    float local_sq_right[block_size];

    for (int thread_id = 0; thread_id < block_size; thread_id++) {
      int64_t global_id = static_cast<int64_t>(block_id) * block_size + thread_id;
      if (global_id >= total_size) break;
      const int8_t exp_full = exp[global_id];
      const int8_t sq_full = sq[global_id];

      const float p_left = p[global_id * 2];
      const float p_right = p[global_id * 2 + 1];
      const float g_left = g[global_id * 2];
      const float g_right = g[global_id * 2 + 1];

      const int8_t exp_left_index = exp_full & bitmask;
      const int8_t sq_left_index = sq_full & bitmask;

      const float p_left_decay = p_left * weight_decay_update;
      const float exp_avg_qscale = exp_qscale[block_id];

      float exp_left = exp_qmap[exp_left_index] * exp_avg_qscale;
      exp_left = beta1 * exp_left + resid_beta1 * g_left;

      float sq_left = sq_qmap[sq_left_index] * sq_qscale[block_id];
      sq_left = beta2 * sq_left + resid_beta2 * (g_left * g_left);

      p[global_id * 2] =
          p_left_decay - (step_size * (exp_left / (std::sqrt(sq_left) / correction2_sqrt + eps)));

      const int8_t exp_right_index = (exp_full >> 4) & bitmask;
      const int8_t sq_right_index = (sq_full >> 4) & bitmask;

      const float p_right_decay = p_right * weight_decay_update;

      float exp_right = exp_qmap[exp_right_index] * exp_avg_qscale;
      exp_right = beta1 * exp_right + resid_beta1 * g_right;

      float sq_right = sq_qmap[sq_right_index] * sq_qscale[block_id];
      sq_right = beta2 * sq_right + resid_beta2 * (g_right * g_right);

      p[global_id * 2 + 1] =
          p_right_decay - (step_size * (exp_right / (std::sqrt(sq_right) / correction2_sqrt + eps)));

      local_exp_left[thread_id] = exp_left;
      local_exp_right[thread_id] = exp_right;
      local_sq_left[thread_id] = sq_left;
      local_sq_right[thread_id] = sq_right;

      absmax_exp = std::fmax(absmax_exp, std::fmax(exp_left, exp_right));
      absmax_sq = std::fmax(absmax_sq, std::fmax(sq_left, sq_right));
    }

    exp_qscale[block_id] = absmax_exp;
    sq_qscale[block_id] = absmax_sq;

    for (int thread_id = 0; thread_id < block_size; thread_id++) {
      int64_t global_id = static_cast<int64_t>(block_id) * block_size + thread_id;
      if (global_id >= total_size) break;

      int8_t local_packed_exp = 0;
      int8_t local_packed_sq = 0;

      const int8_t q_exp_left =
          static_cast<int8_t>(q_mapping(exp_qmap, exp_qmidpt, local_exp_left[thread_id] / absmax_exp));
      const int8_t q_sq_left =
          static_cast<int8_t>(q_mapping(sq_qmap, sq_qmidpt, local_sq_left[thread_id] / absmax_sq));
      local_packed_exp |= (q_exp_left & bitmask);
      local_packed_sq |= (q_sq_left & bitmask);

      const int8_t q_exp_right =
          static_cast<int8_t>(q_mapping(exp_qmap, exp_qmidpt, local_exp_right[thread_id] / absmax_exp));
      const int8_t q_sq_right =
          static_cast<int8_t>(q_mapping(sq_qmap, sq_qmidpt, local_sq_right[thread_id] / absmax_sq));
      local_packed_exp |= (q_exp_right & right_pack_bitmask);
      local_packed_sq |= (q_sq_right & right_pack_bitmask);

      exp[global_id] = local_packed_exp;
      sq[global_id] = local_packed_sq;
    }
  }
}
