#include <hip/hip_runtime.h>

#include <cstdint>

extern "C" __global__ void compute_ssim(const uint32_t dimx,
                                         const uint32_t dimy,
                                         const uint32_t dimz,
                                         const float* __restrict__ fx_in,
                                         const float* __restrict__ fy_in,
                                         const uint32_t gdimx,
                                         const uint32_t gdimy,
                                         const uint32_t gdimz,
                                         const float data_range,
                                         const float cov_norm,
                                         const float K1,
                                         const float K2,
                                         float* __restrict__ out) {
  constexpr int WIN_SIZE = 7;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t count = dimx * dimy * dimz;
  if (idx >= count) return;
  const uint32_t x = idx % dimx;
  const uint32_t y = (idx / dimx) % dimy;
  const uint32_t z = idx / (dimx * dimy);

  float ux = 0.f;
  float uy = 0.f;
  float uxx = 0.f;
  float uyy = 0.f;
  float uxy = 0.f;

  for (int kz = 0; kz < WIN_SIZE; ++kz) {
    for (int ky = 0; ky < WIN_SIZE; ++ky) {
      for (int kx = 0; kx < WIN_SIZE; ++kx) {
        const uint32_t gx = x + kx;
        const uint32_t gy = y + ky;
        const uint32_t gz = z + kz;
        const uint32_t gidx = gx + gy * gdimx + gz * gdimx * gdimy;
        const float fx = fx_in[gidx];
        const float fy = fy_in[gidx];

        ux += fx;
        uy += fy;
        uxx += fx * fx;
        uyy += fy * fy;
        uxy += fx * fy;
      }
    }
  }

  const float w = 1.f / (WIN_SIZE * WIN_SIZE * WIN_SIZE);
  ux *= w;
  uy *= w;
  uxx *= w;
  uyy *= w;
  uxy *= w;

  const float vx = cov_norm * (uxx - ux * ux);
  const float vy = cov_norm * (uyy - uy * uy);
  const float vxy = cov_norm * (uxy - ux * uy);

  const float R = data_range;
  const float C1 = (K1 * R) * (K1 * R);
  const float C2 = (K2 * R) * (K2 * R);

  const float A1 = 2.f * ux * uy + C1;
  const float A2 = 2.f * vxy + C2;
  const float B1 = ux * ux + uy * uy + C1;
  const float B2 = vx + vy + C2;
  const float D = B1 * B2;
  const float S = (A1 * A2) / D;

  out[idx] = S;
}

extern "C" __global__ void reduce_ssim(const float* __restrict__ in,
                                        const uint32_t n,
                                        float* __restrict__ partials) {
  __shared__ float scratch[256];
  const uint32_t tid = threadIdx.x;
  const uint32_t idx = blockIdx.x * blockDim.x + tid;
  scratch[tid] = idx < n ? in[idx] : 0.f;
  __syncthreads();

  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = scratch[0];
  }
}
