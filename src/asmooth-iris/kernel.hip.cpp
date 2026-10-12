#include <hip/hip_runtime.h>

extern "C" __global__ void smoothingFilter(
    int Lx, int Ly,
    int Threshold, int MaxRad,
    const float* __restrict__ Img,
    int* __restrict__ Box,
    float* __restrict__ Norm)
{
  int tid = threadIdx.x;
  int tjd = threadIdx.y;
  int i = blockIdx.x * blockDim.x + tid;
  int j = blockIdx.y * blockDim.y + tjd;
  int stid = tjd * blockDim.x + tid;
  int gtid = j * Lx + i;

  __shared__ float s_Img[1024];

  if (i < Lx && j < Ly)
    s_Img[stid] = Img[gtid];

  __syncthreads();

  if (i < Lx && j < Ly) {
    float sum = 0.f;
    int q = 1;
    int s = q;
    int ksum = 0;

    while (sum < Threshold && q < MaxRad) {
      s = q;
      sum = 0.f;
      ksum = 0;

      for (int ii = -s; ii < s + 1; ii++)
        for (int jj = -s; jj < s + 1; jj++)
          if ((i - s >= 0) && (i + s < Ly) && (j - s >= 0) && (j + s < Lx)) {
            ksum++;
            if (tid - s >= 0 && tid + s < blockDim.x && tjd - s >= 0 && tjd + s < blockDim.y)
              sum += s_Img[stid + ii * blockDim.x + jj];
            else
              sum += Img[gtid + ii * Lx + jj];
          }
      q++;
    }
    Box[gtid] = s;

    for (int ii = -s; ii < s + 1; ii++)
      for (int jj = -s; jj < s + 1; jj++)
        if (ksum != 0)
          atomicAdd(&Norm[gtid + ii * Lx + jj], __fdividef(1.f, (float)ksum));
  }
}

extern "C" __global__ void normalizeFilter(
    int Lx, int Ly,
    float* __restrict__ Img,
    const float* __restrict__ Norm)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < Lx && j < Ly) {
    int gtid = j * Lx + i;
    const float norm = Norm[gtid];
    if (norm != 0) Img[gtid] = __fdividef(Img[gtid], norm);
  }
}

extern "C" __global__ void outFilter(
    int Lx, int Ly,
    const float* __restrict__ Img,
    const int* __restrict__ Box,
    float* __restrict__ Out)
{
  int tid = threadIdx.x;
  int tjd = threadIdx.y;
  int i = blockIdx.x * blockDim.x + tid;
  int j = blockIdx.y * blockDim.y + tjd;
  int stid = tjd * blockDim.x + tid;
  int gtid = j * Lx + i;

  __shared__ float s_Img[1024];

  if (i < Lx && j < Ly)
    s_Img[stid] = Img[gtid];

  __syncthreads();

  if (i < Lx && j < Ly) {
    const int s = Box[gtid];
    float sum = 0.f;
    int ksum = 0;

    for (int ii = -s; ii < s + 1; ii++)
      for (int jj = -s; jj < s + 1; jj++)
        if ((i - s >= 0) && (i + s < Lx) && (j - s >= 0) && (j + s < Ly)) {
          ksum++;
          if (tid - s >= 0 && tid + s < blockDim.x && tjd - s >= 0 && tjd + s < blockDim.y)
            sum += s_Img[stid + ii * blockDim.y + jj];
          else
            sum += Img[gtid + ii * Ly + jj];
        }
    if (ksum != 0) Out[gtid] = __fdividef(sum, (float)ksum);
  }
}
