#include <hip/hip_runtime.h>

#include <stdint.h>

extern "C" __global__ void upsample_forward_kernel(const float* __restrict__ x,
                                                    float* __restrict__ out,
                                                    int B, int C, int H, int W) {
  uint64_t flat_idx = static_cast<uint64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  uint64_t img_size = static_cast<uint64_t>(H) * W;
  if (flat_idx >= static_cast<uint64_t>(B) * C * img_size) return;

  uint64_t H_out = static_cast<uint64_t>(H) * 2;
  uint64_t W_out = static_cast<uint64_t>(W) * 2;
  uint64_t img_out_size = H_out * W_out;
  uint64_t b = flat_idx / (static_cast<uint64_t>(C) * img_size);
  uint64_t c = (flat_idx / img_size) % C;
  uint64_t i = (flat_idx / W) % H;
  uint64_t j = flat_idx % W;
  float x_val = x[b * C * img_size + c * img_size + i * W + j];
  uint64_t offset = b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;
  out[offset] = x_val;
  out[offset + 1] = x_val;
  out[offset + W_out] = x_val;
  out[offset + W_out + 1] = x_val;
}

extern "C" __global__ void upsample_forward_kernel2(const float* __restrict__ x,
                                                     float* __restrict__ out,
                                                     int B, int C, int H, int W) {
  uint64_t in_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t in_y = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t bc = blockIdx.z;
  if (in_x >= static_cast<uint64_t>(W) || in_y >= static_cast<uint64_t>(H)) return;

  uint64_t b = bc / C;
  uint64_t c = bc % C;
  uint64_t H_out = static_cast<uint64_t>(H) * 2;
  uint64_t W_out = static_cast<uint64_t>(W) * 2;
  float val = x[(b * C + c) * static_cast<uint64_t>(H) * W + in_y * W + in_x];
  uint64_t offset = (b * C + c) * H_out * W_out + (in_y * 2) * W_out + (in_x * 2);
  out[offset] = val;
  out[offset + 1] = val;
  out[offset + W_out] = val;
  out[offset + W_out + 1] = val;
}

extern "C" __global__ void upsample_backward_kernel(const float* __restrict__ dout,
                                                     float* __restrict__ dx,
                                                     int B, int C, int H, int W) {
  uint64_t flat_idx = static_cast<uint64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  uint64_t img_in_size = static_cast<uint64_t>(H) * W;
  if (flat_idx >= static_cast<uint64_t>(B) * C * img_in_size) return;

  uint64_t H_out = static_cast<uint64_t>(H) * 2;
  uint64_t W_out = static_cast<uint64_t>(W) * 2;
  uint64_t img_out_size = H_out * W_out;
  uint64_t b = flat_idx / (static_cast<uint64_t>(C) * img_in_size);
  uint64_t c = (flat_idx / img_in_size) % C;
  uint64_t i = (flat_idx / W) % H;
  uint64_t j = flat_idx % W;
  uint64_t offset = b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;
  dx[flat_idx] = dout[offset] + dout[offset + 1] + dout[offset + W_out] +
                 dout[offset + W_out + 1];
}

extern "C" __global__ void upsample_backward_kernel2(const float* __restrict__ dout,
                                                      float* __restrict__ dx,
                                                      int B, int C, int H, int W) {
  uint64_t in_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t in_y = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t bc = blockIdx.z;
  if (in_x >= static_cast<uint64_t>(W) || in_y >= static_cast<uint64_t>(H)) return;

  uint64_t b = bc / C;
  uint64_t c = bc % C;
  uint64_t W_out = static_cast<uint64_t>(W) * 2;
  uint64_t H_out = static_cast<uint64_t>(H) * 2;
  uint64_t offset = b * (static_cast<uint64_t>(C) * H_out * W_out) +
                    c * (H_out * W_out) + (in_y * 2) * W_out + (in_x * 2);
  float grad = dout[offset] + dout[offset + 1] + dout[offset + W_out] +
               dout[offset + W_out + 1];
  dx[(b * C + c) * static_cast<uint64_t>(H) * W + in_y * W + in_x] = grad;
}
