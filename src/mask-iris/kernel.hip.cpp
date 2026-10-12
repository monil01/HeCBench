#include <hip/hip_runtime.h>

#define KERNEL_LOOP(index, range) \
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (range); index += blockDim.x * gridDim.x)

extern "C" __global__ void sequenceMaskKernel_int(int N, int M, int B, const int* in, const int* seq_lengths, int fill_val, int* out)
{
  int total = (B >= 0 ? B : 1) * N * M;
  KERNEL_LOOP(index, total) {
    int k = index % M;
    int j = (index - k) / M % N;
    out[index] = (k >= seq_lengths[j] ? fill_val : in[index]);
  }
}

extern "C" __global__ void windowMaskKernel_int(int N, int M, int B, const int* in, const int* window_centers, int radius, int fill_val, int* out)
{
  int total = (B >= 0 ? B : 1) * N * M;
  KERNEL_LOOP(index, total) {
    int k = index % M;
    int j = (index - k) / M % N;
    out[index] = (k < window_centers[j] - radius || k > window_centers[j] + radius ? fill_val : in[index]);
  }
}

extern "C" __global__ void upperMaskKernel_int(int N, int M, int B, const int* in, int fill_val, int* out)
{
  int total = (B >= 0 ? B : 1) * N * M;
  KERNEL_LOOP(index, total) {
    int k = index % M;
    int j = (index - k) / M % N;
    out[index] = (k > j ? fill_val : in[index]);
  }
}

extern "C" __global__ void lowerMaskKernel_int(int N, int M, int B, const int* in, int fill_val, int* out)
{
  int total = (B >= 0 ? B : 1) * N * M;
  KERNEL_LOOP(index, total) {
    int k = index % M;
    int j = (index - k) / M % N;
    out[index] = (k < j ? fill_val : in[index]);
  }
}

extern "C" __global__ void upperDiagMaskKernel_int(int N, int M, int B, const int* in, int fill_val, int* out)
{
  int total = (B >= 0 ? B : 1) * N * M;
  KERNEL_LOOP(index, total) {
    int k = index % M;
    int j = (index - k) / M % N;
    out[index] = (k >= j ? fill_val : in[index]);
  }
}

extern "C" __global__ void lowerDiagMaskKernel_int(int N, int M, int B, const int* in, int fill_val, int* out)
{
  int total = (B >= 0 ? B : 1) * N * M;
  KERNEL_LOOP(index, total) {
    int k = index % M;
    int j = (index - k) / M % N;
    out[index] = (k <= j ? fill_val : in[index]);
  }
}
