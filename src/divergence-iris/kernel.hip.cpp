#include <hip/hip_runtime.h>

extern "C" __global__ void div_kernel(const float* gv,
                                       const float* Dvv,
                                       float* div,
                                       float* vvtemp,
                                       const float* rmetdet,
                                       int np)
{
  constexpr float rrearth = 1.5683814303638645E-7f;
  int l = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (l < np && j < np) {
    float dudx00 = 0.0f;
    float dvdy00 = 0.0f;
    for (int i = 0; i < np; i++) {
      dudx00 += Dvv[l * np + i] * gv[j * np * 2 + i * 2];
      dvdy00 += Dvv[l * np + i] * gv[i * np * 2 + j * 2 + 1];
    }
    div[j * np + l] = dudx00;
    vvtemp[l * np + j] = dvdy00;
  }
  __syncthreads();

  if (l < np && j < np) {
    div[l * np + j] = (div[l * np + j] + vvtemp[l * np + j]) *
                      (rmetdet[l * np + j] * rrearth);
  }
}
