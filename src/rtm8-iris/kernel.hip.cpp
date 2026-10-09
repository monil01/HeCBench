#include <hip/hip_runtime.h>

#define nx 680
#define ny 134
#define nz 450

__device__ int indexTo1D(int x, int y, int z) { return x + y * nx + z * nx * ny; }

extern "C" __global__ void rtm8(const float* __restrict__ vsq,
                                const float* __restrict__ current_s,
                                const float* __restrict__ current_r,
                                float* __restrict__ next_s,
                                float* __restrict__ next_r,
                                float* __restrict__ image,
                                const float* __restrict__ a, size_t) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
  if ((4 <= x && x < (nx - 4)) && (4 <= y && y < (ny - 4)) &&
      (4 <= z && z < (nz - 4))) {
    float div =
        a[0] * current_s[indexTo1D(x, y, z)] +
        a[1] * (current_s[indexTo1D(x + 1, y, z)] +
                current_s[indexTo1D(x - 1, y, z)] +
                current_s[indexTo1D(x, y + 1, z)] +
                current_s[indexTo1D(x, y - 1, z)] +
                current_s[indexTo1D(x, y, z + 1)] +
                current_s[indexTo1D(x, y, z - 1)]) +
        a[2] * (current_s[indexTo1D(x + 2, y, z)] +
                current_s[indexTo1D(x - 2, y, z)] +
                current_s[indexTo1D(x, y + 2, z)] +
                current_s[indexTo1D(x, y - 2, z)] +
                current_s[indexTo1D(x, y, z + 2)] +
                current_s[indexTo1D(x, y, z - 2)]) +
        a[3] * (current_s[indexTo1D(x + 3, y, z)] +
                current_s[indexTo1D(x - 3, y, z)] +
                current_s[indexTo1D(x, y + 3, z)] +
                current_s[indexTo1D(x, y - 3, z)] +
                current_s[indexTo1D(x, y, z + 3)] +
                current_s[indexTo1D(x, y, z - 3)]) +
        a[4] * (current_s[indexTo1D(x + 4, y, z)] +
                current_s[indexTo1D(x - 4, y, z)] +
                current_s[indexTo1D(x, y + 4, z)] +
                current_s[indexTo1D(x, y - 4, z)] +
                current_s[indexTo1D(x, y, z + 4)] +
                current_s[indexTo1D(x, y, z - 4)]);

    next_s[indexTo1D(x, y, z)] =
        2 * current_s[indexTo1D(x, y, z)] - next_s[indexTo1D(x, y, z)] +
        vsq[indexTo1D(x, y, z)] * div;
    div =
        a[0] * current_r[indexTo1D(x, y, z)] +
        a[1] * (current_r[indexTo1D(x + 1, y, z)] +
                current_r[indexTo1D(x - 1, y, z)] +
                current_r[indexTo1D(x, y + 1, z)] +
                current_r[indexTo1D(x, y - 1, z)] +
                current_r[indexTo1D(x, y, z + 1)] +
                current_r[indexTo1D(x, y, z - 1)]) +
        a[2] * (current_r[indexTo1D(x + 2, y, z)] +
                current_r[indexTo1D(x - 2, y, z)] +
                current_r[indexTo1D(x, y + 2, z)] +
                current_r[indexTo1D(x, y - 2, z)] +
                current_r[indexTo1D(x, y, z + 2)] +
                current_r[indexTo1D(x, y, z - 2)]) +
        a[3] * (current_r[indexTo1D(x + 3, y, z)] +
                current_r[indexTo1D(x - 3, y, z)] +
                current_r[indexTo1D(x, y + 3, z)] +
                current_r[indexTo1D(x, y - 3, z)] +
                current_r[indexTo1D(x, y, z + 3)] +
                current_r[indexTo1D(x, y, z - 3)]) +
        a[4] * (current_r[indexTo1D(x + 4, y, z)] +
                current_r[indexTo1D(x - 4, y, z)] +
                current_r[indexTo1D(x, y + 4, z)] +
                current_r[indexTo1D(x, y - 4, z)] +
                current_r[indexTo1D(x, y, z + 4)] +
                current_r[indexTo1D(x, y, z - 4)]);

    next_r[indexTo1D(x, y, z)] =
        2 * current_r[indexTo1D(x, y, z)] - next_r[indexTo1D(x, y, z)] +
        vsq[indexTo1D(x, y, z)] * div;
    image[indexTo1D(x, y, z)] =
        next_s[indexTo1D(x, y, z)] * next_r[indexTo1D(x, y, z)];
  }
}
