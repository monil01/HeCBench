#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <math.h>

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
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int i = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    max_ = fmaxf(max_, src[i * sliceSize + j]);
  }
  max_ = cg::reduce(warp, max_, cg::greater<float>{});
  float sum = 0;
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    sum += expf(src[i * sliceSize + j] - max_);
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
  }
}
