#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include <cstdint>

namespace {

constexpr int UNROLL = 4;

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

}  // namespace

extern "C" __global__ void fused_dropout_kernel_vec1(
    const float* __restrict__ a, float* __restrict__ b, uint8_t* __restrict__ c,
    uint32_t totalElements, float p, uint64_t seed_first, uint64_t seed_second) {
  const float scale = 1.0f / p;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed_first, idx, seed_second, &state);

  const uint32_t stride = gridDim.x * blockDim.x * UNROLL;
  const uint32_t rounded_size = ((totalElements - 1) / stride + 1) * stride;

  for (uint32_t linearIndex = idx; linearIndex < rounded_size; linearIndex += stride) {
    float4 rand = hiprand_uniform4(&state);
    float src[UNROLL] = {0.0f, 0.0f, 0.0f, 0.0f};
    rand.x = rand.x < p;
    rand.y = rand.y < p;
    rand.z = rand.z < p;
    rand.w = rand.w < p;

    #pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      uint32_t li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) src[ii] = a[li];
    }

    #pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      uint32_t li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        const float keep = (&rand.x)[ii];
        b[li] = src[ii] * keep * scale;
        c[li] = static_cast<uint8_t>(keep);
      }
    }
  }
}

extern "C" __global__ void fused_dropout_kernel_vec2(
    const float* __restrict__ a, float* __restrict__ b, uint8_t* __restrict__ c,
    uint32_t totalElements, float p, uint64_t seed_first, uint64_t seed_second) {
  using LoadT = aligned_vector<float, 2>;
  using MaskLoadT = aligned_vector<uint8_t, 2>;
  const float scale = 1.0f / p;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  bool gridxvec_loop_state = false;

  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed_first, idx, seed_second, &state);

  for (uint32_t linearIndex = idx * 2; linearIndex + 1 < totalElements;
       linearIndex += gridDim.x * blockDim.x * 2) {
    float src[2];
    float4 rand;
    if (!gridxvec_loop_state) {
      rand = hiprand_uniform4(&state);
    } else {
      rand.x = rand.z;
      rand.y = rand.w;
      gridxvec_loop_state = !gridxvec_loop_state;
    }
    rand.x = rand.x < p;
    rand.y = rand.y < p;

    *reinterpret_cast<LoadT*>(src) = *reinterpret_cast<const LoadT*>(&a[linearIndex]);
    float out[2] = {src[0] * rand.x * scale, src[1] * rand.y * scale};
    uint8_t mask[2] = {static_cast<uint8_t>(rand.x), static_cast<uint8_t>(rand.y)};
    *reinterpret_cast<LoadT*>(&b[linearIndex]) = *reinterpret_cast<LoadT*>(out);
    *reinterpret_cast<MaskLoadT*>(&c[linearIndex]) = *reinterpret_cast<MaskLoadT*>(mask);
  }
}

extern "C" __global__ void fused_dropout_kernel_vec4(
    const float* __restrict__ a, float* __restrict__ b, uint8_t* __restrict__ c,
    uint32_t totalElements, float p, uint64_t seed_first, uint64_t seed_second) {
  using LoadT = aligned_vector<float, 4>;
  using MaskLoadT = aligned_vector<uint8_t, 4>;
  const float scale = 1.0f / p;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed_first, idx, seed_second, &state);

  for (uint32_t linearIndex = idx * 4; linearIndex + 3 < totalElements;
       linearIndex += gridDim.x * blockDim.x * 4) {
    float4 rand = hiprand_uniform4(&state);
    rand.x = rand.x < p;
    rand.y = rand.y < p;
    rand.z = rand.z < p;
    rand.w = rand.w < p;

    float src[4];
    *reinterpret_cast<LoadT*>(src) = *reinterpret_cast<const LoadT*>(&a[linearIndex]);
    float out[4] = {
        src[0] * rand.x * scale, src[1] * rand.y * scale,
        src[2] * rand.z * scale, src[3] * rand.w * scale};
    uint8_t mask[4] = {
        static_cast<uint8_t>(rand.x), static_cast<uint8_t>(rand.y),
        static_cast<uint8_t>(rand.z), static_cast<uint8_t>(rand.w)};
    *reinterpret_cast<LoadT*>(&b[linearIndex]) = *reinterpret_cast<LoadT*>(out);
    *reinterpret_cast<MaskLoadT*>(&c[linearIndex]) = *reinterpret_cast<MaskLoadT*>(mask);
  }
}
