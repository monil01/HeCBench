#include <cuda.h>
#include <float.h>

struct alignas(8) vec_2d_float {
  float x;
  float y;
};

struct extrema_pair {
  float min_value;
  float max_value;
  int min_index;
  int max_index;
};

extern "C" __global__ void minmax_kernel(const vec_2d_float* points,
                                          extrema_pair* partials,
                                          const int n)
{
  __shared__ float s_min_value[256];
  __shared__ float s_max_value[256];
  __shared__ int s_min_index[256];
  __shared__ int s_max_index[256];

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  float min_value = FLT_MAX;
  float max_value = -FLT_MAX;
  int min_index = -1;
  int max_index = -1;

  if (idx < n) {
    const float value = points[idx].x * points[idx].x + points[idx].y * points[idx].y;
    min_value = value;
    max_value = value;
    min_index = idx;
    max_index = idx;
  }

  s_min_value[tid] = min_value;
  s_max_value[tid] = max_value;
  s_min_index[tid] = min_index;
  s_max_index[tid] = max_index;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (s_min_index[tid + stride] >= 0 && s_min_value[tid + stride] < s_min_value[tid]) {
        s_min_value[tid] = s_min_value[tid + stride];
        s_min_index[tid] = s_min_index[tid + stride];
      }
      if (s_max_index[tid + stride] >= 0 && s_max_value[tid + stride] > s_max_value[tid]) {
        s_max_value[tid] = s_max_value[tid + stride];
        s_max_index[tid] = s_max_index[tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x].min_value = s_min_value[0];
    partials[blockIdx.x].max_value = s_max_value[0];
    partials[blockIdx.x].min_index = s_min_index[0];
    partials[blockIdx.x].max_index = s_max_index[0];
  }
}
