#include <cuda.h>

__device__ __forceinline__ void clip_plus(int clip, int n, int& plus)
{
  if (plus >= n) plus = clip ? n - 1 : plus - n;
}

__device__ __forceinline__ void clip_minus(int clip, int n, int& minus)
{
  if (minus < 0) minus = clip ? 0 : minus + n;
}

template <typename T>
__device__ void relextrema_1d_impl(int n,
                                   int order,
                                   int clip,
                                   const T* __restrict__ inp,
                                   unsigned char* __restrict__ results)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int tid = tx; tid < n; tid += stride) {
    const T data = inp[tid];
    bool temp = true;
    for (int o = 1; o <= order; o++) {
      int plus = tid + o;
      int minus = tid - o;
      clip_plus(clip, n, plus);
      clip_minus(clip, n, minus);
      temp &= data > inp[plus];
      temp &= data >= inp[minus];
    }
    results[tid] = temp ? 1 : 0;
  }
}

template <typename T>
__device__ void relextrema_2d_impl(int in_x,
                                   int in_y,
                                   int order,
                                   int clip,
                                   int axis,
                                   const T* __restrict__ inp,
                                   unsigned char* __restrict__ results)
{
  const int ty = blockIdx.x * blockDim.x + threadIdx.x;
  const int tx = blockIdx.y * blockDim.y + threadIdx.y;
  if ((tx < in_y) && (ty < in_x)) {
    const int tid = tx * in_x + ty;
    const T data = inp[tid];
    bool temp = true;
    for (int o = 1; o <= order; o++) {
      int plus;
      int minus;
      if (axis == 0) {
        plus = tx + o;
        minus = tx - o;
        clip_plus(clip, in_y, plus);
        clip_minus(clip, in_y, minus);
        plus = plus * in_x + ty;
        minus = minus * in_x + ty;
      } else {
        plus = ty + o;
        minus = ty - o;
        clip_plus(clip, in_x, plus);
        clip_minus(clip, in_x, minus);
        plus = tx * in_x + plus;
        minus = tx * in_x + minus;
      }
      temp &= data > inp[plus];
      temp &= data >= inp[minus];
    }
    results[tid] = temp ? 1 : 0;
  }
}

#define DEFINE_EXTREMA_KERNELS(TYPE, NAME)                                      \
  extern "C" __global__ void relextrema_1d_##NAME(                             \
      int n, int order, int clip, const TYPE* inp, unsigned char* results)      \
  {                                                                            \
    relextrema_1d_impl<TYPE>(n, order, clip, inp, results);                    \
  }                                                                            \
  extern "C" __global__ void relextrema_2d_##NAME(                             \
      int in_x, int in_y, int order, int clip, int axis,                       \
      const TYPE* inp, unsigned char* results)                                 \
  {                                                                            \
    relextrema_2d_impl<TYPE>(in_x, in_y, order, clip, axis, inp, results);     \
  }

DEFINE_EXTREMA_KERNELS(int, int)
DEFINE_EXTREMA_KERNELS(long, long)
DEFINE_EXTREMA_KERNELS(float, float)
DEFINE_EXTREMA_KERNELS(double, double)
