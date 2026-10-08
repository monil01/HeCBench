#include <hip/hip_runtime.h>

template <typename T, typename StrideType>
__device__ void cross_kernel_impl(
    int numel,
    T* out,
    const T* x1,
    const T* x2,
    StrideType ostride,
    StrideType x1stride,
    StrideType x2stride)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < numel; i += blockDim.x * gridDim.x) {
    T* out_row = out + 3 * i;
    const T* x1_row = x1 + 3 * i;
    const T* x2_row = x2 + 3 * i;

    const T val0 = (x1_row[1 * x1stride] * x2_row[2 * x2stride] -
                    x1_row[2 * x1stride] * x2_row[1 * x2stride]);
    const T val1 = (x1_row[2 * x1stride] * x2_row[0 * x2stride] -
                    x1_row[0 * x1stride] * x2_row[2 * x2stride]);
    const T val2 = (x1_row[0 * x1stride] * x2_row[1 * x2stride] -
                    x1_row[1 * x1stride] * x2_row[0 * x2stride]);

    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

template <typename T, typename StrideType>
__device__ void cross2_kernel_impl(
    int numel,
    T* out,
    const T* x1,
    const T* x2,
    StrideType ostride,
    StrideType x1stride,
    StrideType x2stride)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < numel; i += blockDim.x * gridDim.x) {
    T* out_row = out + 3 * i;
    const T* x1_row = x1 + 3 * i;
    const T* x2_row = x2 + 3 * i;

    const T x1_c0 = x1_row[0 * x1stride];
    const T x1_c1 = x1_row[1 * x1stride];
    const T x1_c2 = x1_row[2 * x1stride];
    const T x2_c0 = x2_row[0 * x2stride];
    const T x2_c1 = x2_row[1 * x2stride];
    const T x2_c2 = x2_row[2 * x2stride];

    const T val0 = x1_c1 * x2_c2 - x1_c2 * x2_c1;
    const T val1 = x1_c2 * x2_c0 - x1_c0 * x2_c2;
    const T val2 = x1_c0 * x2_c1 - x1_c1 * x2_c0;

    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

template <typename T>
__device__ void cross3_kernel_impl(
    int numel,
    T* out,
    const T* x1,
    const T* x2)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < numel; i += blockDim.x * gridDim.x) {
    T* out_row = out + 3 * i;
    const T* x1_row = x1 + 3 * i;
    const T* x2_row = x2 + 3 * i;

    const T x1_c0 = x1_row[0];
    const T x1_c1 = x1_row[1];
    const T x1_c2 = x1_row[2];
    const T x2_c0 = x2_row[0];
    const T x2_c1 = x2_row[1];
    const T x2_c2 = x2_row[2];

    const T val0 = x1_c1 * x2_c2 - x1_c2 * x2_c1;
    const T val1 = x1_c2 * x2_c0 - x1_c0 * x2_c2;
    const T val2 = x1_c0 * x2_c1 - x1_c1 * x2_c0;

    out_row[0] = val0;
    out_row[1] = val1;
    out_row[2] = val2;
  }
}

extern "C" __global__ void cross_kernel_float(
    int numel,
    float* out,
    const float* x1,
    const float* x2,
    int ostride,
    int x1stride,
    int x2stride)
{
  cross_kernel_impl(numel, out, x1, x2, ostride, x1stride, x2stride);
}

extern "C" __global__ void cross2_kernel_float(
    int numel,
    float* out,
    const float* x1,
    const float* x2,
    int ostride,
    int x1stride,
    int x2stride)
{
  cross2_kernel_impl(numel, out, x1, x2, ostride, x1stride, x2stride);
}

extern "C" __global__ void cross3_kernel_float(
    int numel,
    float* out,
    const float* x1,
    const float* x2)
{
  cross3_kernel_impl(numel, out, x1, x2);
}

extern "C" __global__ void cross_kernel_double(
    int numel,
    double* out,
    const double* x1,
    const double* x2,
    int ostride,
    int x1stride,
    int x2stride)
{
  cross_kernel_impl(numel, out, x1, x2, ostride, x1stride, x2stride);
}

extern "C" __global__ void cross2_kernel_double(
    int numel,
    double* out,
    const double* x1,
    const double* x2,
    int ostride,
    int x1stride,
    int x2stride)
{
  cross2_kernel_impl(numel, out, x1, x2, ostride, x1stride, x2stride);
}

extern "C" __global__ void cross3_kernel_double(
    int numel,
    double* out,
    const double* x1,
    const double* x2)
{
  cross3_kernel_impl(numel, out, x1, x2);
}
