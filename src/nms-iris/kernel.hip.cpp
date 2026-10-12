#include <hip/hip_runtime.h>

#define MAX_DETECTIONS 4096
#define N_PARTITIONS 32

extern "C" __global__ void generate_nms_bitmap(const float4* rects,
                                                unsigned char* nmsbitmap,
                                                const float othreshold)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= MAX_DETECTIONS || j >= MAX_DETECTIONS) return;

  if (rects[i].w < rects[j].w) {
    float area = (rects[j].z + 1.0f) * (rects[j].z + 1.0f);
    float w = fmaxf(0.0f, fminf(rects[i].x + rects[i].z, rects[j].x + rects[j].z) -
                              fmaxf(rects[i].x, rects[j].x) + 1.0f);
    float h = fmaxf(0.0f, fminf(rects[i].y + rects[i].z, rects[j].y + rects[j].z) -
                              fmaxf(rects[i].y, rects[j].y) + 1.0f);
    nmsbitmap[i * MAX_DETECTIONS + j] =
        (((w * h) / area) < othreshold) && (rects[j].z != 0);
  }
}

__device__ __forceinline__ void compute_nms_point_mask(unsigned char* pointsbitmap,
                                                       int cond)
{
  *pointsbitmap = __syncthreads_and(cond);
}

extern "C" __global__ void reduce_nms_bitmap(unsigned char* nmsbitmap,
                                              unsigned char* pointsbitmap,
                                              int ndetections)
{
  const int row = blockIdx.x;
  if (row >= ndetections) return;
  int idx = row * MAX_DETECTIONS + threadIdx.x;

  compute_nms_point_mask(&pointsbitmap[row], nmsbitmap[idx]);
  for (int i = 0; i < (N_PARTITIONS - 1); i++) {
    idx += MAX_DETECTIONS / N_PARTITIONS;
    compute_nms_point_mask(&pointsbitmap[row], pointsbitmap[row] && nmsbitmap[idx]);
  }
}
