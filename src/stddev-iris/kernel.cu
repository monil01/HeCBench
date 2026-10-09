#include <cuda.h>
#include <math.h>

extern "C" __global__ void sampleKernel_float_int(float* std, int D, int N)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < D) std[i] = sqrtf(std[i] / N);
}

extern "C" __global__ void sopKernel_float_int(
    float* __restrict__ std,
    const float* __restrict__ data,
    int D,
    int N)
{
  const int TPB = 256;
  const int ColsPerBlk = 32;
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  const int thisColId = threadIdx.x % ColsPerBlk;
  const int thisRowId = threadIdx.x / ColsPerBlk;
  const int colId = thisColId + static_cast<int>(blockIdx.y) * ColsPerBlk;
  const int rowId = thisRowId + static_cast<int>(blockIdx.x) * RowsPerBlkPerIter;
  float thread_data = 0.0f;
  const int stride = RowsPerBlkPerIter * static_cast<int>(gridDim.x);

  for (int i = rowId; i < N; i += stride) {
    const float val = (colId < D) ? data[static_cast<size_t>(i) * D + colId] : 0.0f;
    thread_data += val * val;
  }

  __shared__ float sstd[ColsPerBlk];
  if (threadIdx.x < ColsPerBlk) sstd[threadIdx.x] = 0.0f;
  __syncthreads();

  atomicAdd(sstd + thisColId, thread_data);
  __syncthreads();

  if (threadIdx.x < ColsPerBlk) atomicAdd(std + colId, sstd[thisColId]);
}
