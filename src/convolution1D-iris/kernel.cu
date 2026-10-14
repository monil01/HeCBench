#include <cuda_runtime.h>

#include <cstdint>

// Fixed capacity replaces the native dynamic shared allocation. Block sizes
// and halo loading retain the CUDA algorithm; mask storage is an IRIS DMem.
template <typename T, bool Cache>
__device__ void tiled_body(const T* in, T* out, int width, int mw, const T* mask) {
  __shared__ T tile[1024 + 10 - 1];
  int t = threadIdx.x, bs = blockDim.x, i = blockIdx.x * bs + t;
  int n = mw / 2;
  if (Cache) tile[t] = i < width ? in[i] : T(0);
  else {
    if (t >= bs - n) {
      int left = (int(blockIdx.x) - 1) * bs + t;
      tile[t - (bs - n)] = left < 0 ? T(0) : in[left];
    }
    tile[n + t] = i < width ? in[i] : T(0);
    if (t < n) {
      int right = (blockIdx.x + 1) * bs + t;
      tile[t + bs + n] = right >= width ? T(0) : in[right];
    }
  }
  __syncthreads();
  if (i >= width) return;
  T sum = 0;
  for (int j = 0; j < mw; ++j) {
    int idx = i - n + j;
    if (!Cache) sum += tile[t + j] * mask[j];
    else if (idx >= 0 && idx < width)
      sum += (idx >= int(blockIdx.x) * bs && idx < int(blockIdx.x + 1) * bs
              ? tile[t + j - n] : in[idx]) * mask[j];
  }
  out[i] = sum;
}
#define VARIANTS(T, S) \
extern "C" __global__ void conv1d_##S##_tiled(const T* a,T* b,int n,int m,const T* mask) { tiled_body<T,false>(a,b,n,m,mask); } \
extern "C" __global__ void conv1d_##S##_caching(const T* a,T* b,int n,int m,const T* mask) { tiled_body<T,true>(a,b,n,m,mask); }
VARIANTS(double, f64)
VARIANTS(float, f32)
VARIANTS(int16_t, i16)

template <typename T>
__device__ void conv1d_body(const T* __restrict__ in, T* __restrict__ out,
                            int input_width, int mask_width,
                            const T* __restrict__ mask) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= input_width) return;
  T s = 0;
  const int start = i - mask_width / 2;
  for (int j = 0; j < mask_width; j++) {
    if (start + j >= 0 && start + j < input_width) {
      s += in[start + j] * mask[j];
    }
  }
  out[i] = s;
}

extern "C" __global__ void conv1d_f64(const double* __restrict__ in,
                                       double* __restrict__ out,
                                       int input_width,
                                       int mask_width,
                                       const double* __restrict__ mask) {
  conv1d_body(in, out, input_width, mask_width, mask);
}

extern "C" __global__ void conv1d_f32(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int input_width,
                                       int mask_width,
                                       const float* __restrict__ mask) {
  conv1d_body(in, out, input_width, mask_width, mask);
}

extern "C" __global__ void conv1d_i16(const int16_t* __restrict__ in,
                                       int16_t* __restrict__ out,
                                       int input_width,
                                       int mask_width,
                                       const int16_t* __restrict__ mask) {
  conv1d_body(in, out, input_width, mask_width, mask);
}
