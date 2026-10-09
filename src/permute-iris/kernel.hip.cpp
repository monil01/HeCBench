#include <hip/hip_runtime.h>

extern "C" __global__ void permute_kernel(float* __restrict__ out,
                                           const float* __restrict__ inp,
                                           int B, int T, int NH, int d)
{
  float* Q = out;
  int C = NH * d;
  float* K = out + B * T * C;
  float* V = out + 2 * B * T * C;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B * C * T) {
    int b = idx / (C * T);
    int rest = idx % (C * T);
    int nh_ = rest / (T * d);
    rest = rest % (T * d);
    int n = rest / d;
    int d_ = rest % d;

    int inp_idx = (b * T * 3 * C) + (n * 3 * C) + (nh_ * d) + d_;
    Q[idx] = inp[inp_idx];
    K[idx] = inp[inp_idx + C];
    V[idx] = inp[inp_idx + 2 * C];
  }
}
