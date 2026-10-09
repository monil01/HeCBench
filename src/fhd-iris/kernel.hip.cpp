#include <hip/hip_runtime.h>

typedef struct {
  float x, y, z;
} kdata;

constexpr float kPi = 3.14159265358979323846f;

extern "C" __global__ void cmpfhd(
    const kdata* __restrict__ k, const float* __restrict__ rmu,
    const float* __restrict__ imu, float* __restrict__ rfhd,
    float* __restrict__ ifhd, const float* __restrict__ x,
    const float* __restrict__ y, const float* __restrict__ z,
    const int samples, const int voxels) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < samples) {
    float xn = x[n], yn = y[n], zn = z[n];
    float rfhdn = rfhd[n], ifhdn = ifhd[n];
    for (int m = 0; m < voxels; m++) {
      float e = 2.f * kPi * (k[m].x * xn + k[m].y * yn + k[m].z * zn);
      float c = __cosf(e);
      float s = __sinf(e);
      rfhdn += rmu[m] * c - imu[m] * s;
      ifhdn += imu[m] * c + rmu[m] * s;
    }
    rfhd[n] = rfhdn, ifhd[n] = ifhdn;
  }
}
