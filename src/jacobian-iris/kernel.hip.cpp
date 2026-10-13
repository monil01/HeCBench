#include <hip/hip_runtime.h>
#include <cmath>
template <typename T>
__device__ __forceinline__ T dev_exp(T v);
template <>
__device__ __forceinline__ float dev_exp<float>(float v) { return expf(v); }
template <>
__device__ __forceinline__ double dev_exp<double>(double v) { return exp(v); }

// k1: one block per sample. Softmax is computed cooperatively into shared
// memory, then the D x D Jacobian is written with a flat, coalesced grid-stride
// loop over the dim*dim output entries.
template <typename T>
__device__ void softmax_jacobian_k1(
    const T* __restrict__ input,
          T* __restrict__ jac,
    int dim)
{
  __shared__ __align__(16) unsigned char smem_raw[(8192 + 1024) * sizeof(T)];
  T* y   = reinterpret_cast<T*>(smem_raw);   // [dim]
  T* red = y + dim;                          // [blockDim.x]

  const int b   = blockIdx.x;
  const int tid = threadIdx.x;
  const int nt  = blockDim.x;
  const T*  x   = input + (size_t)b * dim;
  T*        J   = jac   + (size_t)b * dim * dim;

  // max reduction
  T lmax = -INFINITY;
  for (int i = tid; i < dim; i += nt) lmax = x[i] > lmax ? x[i] : lmax;
  red[tid] = lmax;
  __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] = max(red[tid] , red[tid + s]);
    __syncthreads();
  }
  const T maxv = red[0];
  __syncthreads();

  // exp and sum reduction
  T lsum = 0;
  for (int i = tid; i < dim; i += nt) {
    const T e = dev_exp<T>(x[i] - maxv);
    y[i] = e;
    lsum += e;
  }
  red[tid] = lsum;
  __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const T inv = (T)1 / red[0];
  __syncthreads();

  for (int i = tid; i < dim; i += nt) y[i] *= inv;
  __syncthreads();

  // Jacobian: J[i*dim + j] = y_i * (delta_ij - y_j)
  const size_t total = (size_t)dim * dim;
  for (size_t idx = tid; idx < total; idx += nt) {
    const int i = idx / dim;
    const int j = idx - (size_t)i * dim;
    J[idx] = y[i] * ((i == j ? (T)1 : (T)0) - y[j]);
  }
}

// Warp/wavefront-level primitives -------------------------------------------
// Full-wavefront reductions via shuffle; no shared memory / __syncthreads
// needed within a wavefront. Uses the runtime `warpSize` builtin so the same
// code is correct for both 32- and 64-lane wavefronts.
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T v) {
  for (int o = warpSize >> 1; o > 0; o >>= 1)
    v = max(v, __shfl_down(v, o));
  return v;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
  for (int o = warpSize >> 1; o > 0; o >>= 1)
    v += __shfl_down(v, o);
  return v;
}

// 4-wide vector wrapper for widened (128-bit for float) loads/stores.
template <typename T>
struct alignas(4 * sizeof(T)) Vec4 { T v[4]; };

// k2: one block per sample, aggressively wavefront-centric. The softmax
// max/sum reductions are done with shuffle instructions inside each wavefront
// and only the per-wave partials touch shared memory. The normalized softmax
// vector y is staged in shared memory and the (memory-bound) D x D output is
// streamed with 128-bit vectorized, coalesced stores. All lane/wave arithmetic
// uses `warpSize`, so it runs unchanged on wave32 and wave64 hardware.
template <typename T>
__device__ void softmax_jacobian_k2(
    const T* __restrict__ input,
          T* __restrict__ jac,
    int dim)
{
  // 16-byte alignment lets us reinterpret shared y as Vec4<float> safely.
  __shared__ __align__(16) unsigned char smem_raw[(8192 + 1024) * sizeof(T)];
  T* y     = reinterpret_cast<T*>(smem_raw);   // [dim]
  T* wpart = y + dim;                          // [num_waves] wavefront partials

  const int b        = blockIdx.x;
  const int tid      = threadIdx.x;
  const int nt       = blockDim.x;
  const int lane     = tid & (warpSize - 1);
  const int warp     = tid / warpSize;
  const int nwarps   = (nt + warpSize - 1) / warpSize;
  const T*  x        = input + (size_t)b * dim;
  T*        J        = jac   + (size_t)b * dim * dim;

  // ---- max reduction: wave shuffle, then one wave combines the partials ----
  T lmax = -INFINITY;
  for (int i = tid; i < dim; i += nt) lmax = x[i] > lmax ? x[i] : lmax;
  lmax = warp_reduce_max<T>(lmax);
  if (lane == 0) wpart[warp] = lmax;
  __syncthreads();
  if (warp == 0) {
    T v = (lane < nwarps) ? wpart[lane] : (T)-INFINITY;
    v = warp_reduce_max<T>(v);
    if (lane == 0) wpart[0] = v;
  }
  __syncthreads();
  const T maxv = wpart[0];

  // ---- exp + sum reduction: same wave-centric pattern ----------------------
  T lsum = 0;
  for (int i = tid; i < dim; i += nt) {
    const T e = dev_exp<T>(x[i] - maxv);
    y[i] = e;
    lsum += e;
  }
  lsum = warp_reduce_sum<T>(lsum);
  if (lane == 0) wpart[warp] = lsum;
  __syncthreads();
  if (warp == 0) {
    T v = (lane < nwarps) ? wpart[lane] : (T)0;
    v = warp_reduce_sum<T>(v);
    if (lane == 0) wpart[0] = v;
  }
  __syncthreads();
  const T inv = (T)1 / wpart[0];

  // normalize softmax in shared memory
  for (int i = tid; i < dim; i += nt) y[i] *= inv;
  __syncthreads();

  // ---- Jacobian: J[i*dim + j] = y_i * (delta_ij - y_j) ---------------------
  if ((dim & 3) == 0) {
    const int quads_per_row  = dim >> 2;
    const size_t total_quads = (size_t)dim * (size_t)quads_per_row;
    const Vec4<T>* y4 = reinterpret_cast<const Vec4<T>*>(y);
    Vec4<T>*       J4 = reinterpret_cast<Vec4<T>*>(J);
    for (size_t q = tid; q < total_quads; q += nt) {
      const int i    = (int)(q / (size_t)quads_per_row);
      const int qcol = (int)(q - (size_t)i * quads_per_row);
      const int j0   = qcol << 2;
      const T yi = y[i];
      const Vec4<T> yj = y4[qcol];
      Vec4<T> out;
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        const T d = (i == j0 + k) ? (T)1 : (T)0;
        out.v[k] = yi * (d - yj.v[k]);
      }
      J4[q] = out;
    }
  } else {
    // Scalar fallback for dims not divisible by 4.
    const size_t total = (size_t)dim * dim;
    for (size_t idx = tid; idx < total; idx += nt) {
      const int i = (int)(idx / dim);
      const int j = (int)(idx - (size_t)i * dim);
      J[idx] = y[i] * ((i == j ? (T)1 : (T)0) - y[j]);
    }
  }
}


extern "C" __global__ void jacobian_k1(const float* input,float* jac,int dim) { softmax_jacobian_k1<float>(input,jac,dim); }
extern "C" __global__ void jacobian_k2(const float* input,float* jac,int dim) { softmax_jacobian_k2<float>(input,jac,dim); }
