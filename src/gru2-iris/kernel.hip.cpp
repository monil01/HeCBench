#include <hip/hip_runtime.h>
__device__ __forceinline__ float device_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Store GRU gates in reset, update, new order:
// weight_ih_l[k] = (W_ir | W_iz | W_in), weight_hh_l[k] = (W_hr | W_hz | W_hn).
// W_ih rows are padded to max(input_size, hidden_size) so all layers share one stride.
__device__ __forceinline__ size_t device_wih_offset(
    size_t layer,
    int gate,
    int hidden_idx,
    int input_idx,
    int hidden_size,
    int max_input_size) {
  return ((layer * 3 + gate) * hidden_size + hidden_idx) *
      max_input_size + input_idx;
}

__device__ __forceinline__ size_t device_whh_offset(
    size_t layer,
    int gate,
    int hidden_idx,
    int prev_hidden_idx,
    int hidden_size) {
  return ((layer * 3 + gate) * hidden_size + hidden_idx) *
      hidden_size + prev_hidden_idx;
}

__device__ __forceinline__ size_t device_bias_offset(
    size_t layer,
    int gate,
    int hidden_idx,
    int hidden_size) {
  return (layer * 3 + gate) * hidden_size + hidden_idx;
}

extern "C" __global__ void fused_gru_layer_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h_prev,
    const float* __restrict__ w_ih,
    const float* __restrict__ w_hh,
    const float* __restrict__ b_ih,
    const float* __restrict__ b_hh,
    float* __restrict__ layer_out,
    float* __restrict__ h_next,
    float* __restrict__ output_t,
    int layer,
    int batch_size,
    int input_size,
    int hidden_size,
    int max_input_size, size_t input_offset, size_t hidden_offset,
    size_t output_offset, int last_layer) {
  x += input_offset; h_prev += hidden_offset; h_next += hidden_offset;
  output_t += output_offset;
  const int warp_size = warpSize;
  const int hidden_tile = blockDim.x / warp_size;
  const int b = blockIdx.x;
  const int local_h = threadIdx.x / warp_size;
  const int lane = threadIdx.x % warp_size;
  const int h = blockIdx.y * hidden_tile + local_h;
  if (b >= batch_size || h >= hidden_size) {
    return;
  }

  // One warp/wavefront owns one hidden element. Lanes cooperate on the dot
  // products, so the code works for both 32-lane and 64-lane HIP wavefronts.
  const float* h_row = h_prev + (size_t)b * hidden_size;
  float input_gate[3];
  float hidden_gate[3];
  for (int gate = 0; gate < 3; ++gate) {
    float input_sum = 0.0f;
    for (int k = lane; k < input_size; k += warp_size) {
      input_sum += x[(size_t)b * input_size + k] *
          w_ih[device_wih_offset(layer, gate, h, k, hidden_size, max_input_size)];
    }

    float hidden_sum = 0.0f;
    for (int k = lane; k < hidden_size; k += warp_size) {
      hidden_sum += h_row[k] *
          w_hh[device_whh_offset(layer, gate, h, k, hidden_size)];
    }

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      input_sum += __shfl_down(input_sum, offset, warp_size);
      hidden_sum += __shfl_down(hidden_sum, offset, warp_size);
    }

    if (lane == 0) {
      input_gate[gate] =
          input_sum + b_ih[device_bias_offset(layer, gate, h, hidden_size)];
      hidden_gate[gate] =
          hidden_sum + b_hh[device_bias_offset(layer, gate, h, hidden_size)];
    }
  }

  if (lane == 0) {
    const size_t idx = (size_t)b * hidden_size + h;
    const float hx = h_prev[idx];
    const float r = device_sigmoid(input_gate[0] + hidden_gate[0]);
    const float z = device_sigmoid(input_gate[1] + hidden_gate[1]);
    const float n = tanhf(input_gate[2] + r * hidden_gate[2]);
    const float ht = n + z * (hx - n);
    h_next[idx] = ht;
    layer_out[idx] = ht;
    if (last_layer) {
      output_t[idx] = ht;
    }
  }
}
