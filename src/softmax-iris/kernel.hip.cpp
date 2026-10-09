#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <math.h>

#define BLOCK_SIZE 256

extern "C" __global__ void softMax(
    const int numSlice,
    const int sliceSize,
    const float* src,
    float* dest)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = 0; j < sliceSize; j++) {
    max_ = fmaxf(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) {
    sum += expf(src[i * sliceSize + j] - max_);
  }
  for (int j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
  }
}

extern "C" __global__ void softMax2(
    const int numSlice,
    const int sliceSize,
    const float* src,
    float* dest)
{
#if defined(__GFX8__) || defined(__GFX9__)
#define WarpSize 64
#else
#define WarpSize 32
#endif
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(block);
  int i = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    max_ = fmaxf(max_, src[i * sliceSize + j]);
  }
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
    max_ = fmaxf(max_, warp.shfl_xor(max_, offset));
  }
  float sum = 0;
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    sum += expf(src[i * sliceSize + j] - max_);
  }
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
    sum += warp.shfl_xor(sum, offset);
  }
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
  }
}
