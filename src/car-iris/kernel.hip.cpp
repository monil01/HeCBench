#include <hip/hip_runtime.h>

typedef struct {
  int output_dim_b;
  int output_dim_c;
  int output_dim_h;
  int output_dim_w;
  int kernel_size;
  int image_w;
  int image_h;
} params;

extern "C" __global__ void car(const float* __restrict__ img,
                                const float* __restrict__ kernels,
                                const float* __restrict__ offsets_h,
                                const float* __restrict__ offsets_v,
                                float* __restrict__ output,
                                const params p,
                                const int offset_unit,
                                const int padding,
                                const size_t n)
{
  size_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_idx >= n) return;

  const int dim_b = p.output_dim_b;
  const int dim_c = p.output_dim_c;
  const int dim_h = p.output_dim_h;
  const int dim_w = p.output_dim_w;
  const int kernels_size = p.kernel_size;
  const int img_w = p.image_w;
  const int img_h = p.image_h;

  const size_t vol_size = static_cast<size_t>(dim_c) * dim_h * dim_w;
  const size_t img_size = static_cast<size_t>(dim_h) * dim_w;

  const int idb = (global_idx / vol_size) % dim_b;
  const int idc = (global_idx / img_size) % dim_c;
  const int idy = (global_idx / dim_w) % dim_h;
  const int idx = global_idx % dim_w;

  const int k_size = static_cast<int>(sqrtf(static_cast<float>(kernels_size)));
  const int w = img_w - 2 * padding;
  const int h = img_h - 2 * padding;

  float result = 0;
  for (int k_y = 0; k_y < k_size; ++k_y) {
    for (int k_x = 0; k_x < k_size; ++k_x) {
      const size_t off_idx = static_cast<size_t>(idb) * k_size * k_size * dim_w * dim_h
                           + static_cast<size_t>(k_size * k_y + k_x) * dim_w * dim_h
                           + static_cast<size_t>(idy) * dim_w + idx;
      const float offset_h = offsets_h[off_idx] * offset_unit;
      const float offset_v = offsets_v[off_idx] * offset_unit;
      const float p_x = static_cast<float>(idx + 0.5f) / dim_w * w + k_x + offset_h - 0.5f;
      const float p_y = static_cast<float>(idy + 0.5f) / dim_h * h + k_y + offset_v - 0.5f;
      const float alpha = p_x - floorf(p_x);
      const float beta = p_y - floorf(p_y);
      const int xL = max(min(static_cast<int>(floorf(p_x)), w + 2 * padding - 1), 0);
      const int xR = max(min(xL + 1, w + 2 * padding - 1), 0);
      const int yT = max(min(static_cast<int>(floorf(p_y)), h + 2 * padding - 1), 0);
      const int yB = max(min(yT + 1, h + 2 * padding - 1), 0);
      const size_t img_base = static_cast<size_t>(idb) * dim_c * img_w * img_h
                            + static_cast<size_t>(idc) * img_w * img_h;
      float val = (1.f - alpha) * (1.f - beta) * img[img_base + static_cast<size_t>(yT) * img_w + xL];
      val += alpha * (1.f - beta) * img[img_base + static_cast<size_t>(yT) * img_w + xR];
      val += (1.f - alpha) * beta * img[img_base + static_cast<size_t>(yB) * img_w + xL];
      val += alpha * beta * img[img_base + static_cast<size_t>(yB) * img_w + xR];
      result += val * kernels[off_idx];
    }
  }
  output[static_cast<size_t>(idb) * dim_c * dim_w * dim_h
       + static_cast<size_t>(idc) * dim_w * dim_h
       + static_cast<size_t>(idy) * dim_w + idx] = result;
}
