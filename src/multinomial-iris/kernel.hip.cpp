#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#define GPU_NUM_THREADS 256

extern "C" __global__ void sampleMultinomialOnce(
    int* dest,
    int distributions,
    int categories,
    const float* __restrict__ sampled,
    const float* __restrict__ dist,
    int stride_dist,
    int stride_categories)
{
  __shared__ float smem[GPU_NUM_THREADS];
  __shared__ bool found;
  __shared__ int foundPos;

  typedef hipcub::BlockReduce<float, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const float accZero = 0.0f;
  const float zero = 0.0f;

  for (int curDist = blockIdx.x; curDist < distributions; curDist += gridDim.x) {
    float sum = accZero;
    for (int cat = threadIdx.x; cat < categories; cat += blockDim.x) {
      float val = dist[curDist * stride_dist + cat * stride_categories];
      sum += val;
    }

    sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
      foundPos = 0;
      smem[0] = sum;
      smem[1] = sampled[curDist];
    }
    __syncthreads();

    sum = smem[0];
    const float sample = smem[1];
    __syncthreads();

    if (sum == accZero) {
      if (threadIdx.x == 0) {
        dest[curDist] = 0;
      }
      continue;
    }

    const int chunks = (categories + static_cast<int>(blockDim.x) - 1) / static_cast<int>(blockDim.x);
    float prevHighProb = accZero;
    found = false;

    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      const int cat = chunk * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);

      const float dist_val = cat < categories
          ? dist[curDist * stride_dist + cat * stride_categories] / sum
          : accZero;

      smem[threadIdx.x] = dist_val;
      __syncthreads();

      for (int offset = 1; offset < static_cast<int>(blockDim.x); offset *= 2) {
        float val = accZero;
        if (threadIdx.x >= static_cast<unsigned int>(offset)) {
          val = smem[threadIdx.x - offset] + smem[threadIdx.x];
        }
        __syncthreads();
        if (threadIdx.x >= static_cast<unsigned int>(offset)) {
          smem[threadIdx.x] = val;
        }
        __syncthreads();
      }

      const float curBucket = smem[threadIdx.x] + prevHighProb;
      const float prevBucket = threadIdx.x == 0 ? prevHighProb : smem[threadIdx.x - 1] + prevHighProb;
      const bool inBucket =
          (cat < categories) &&
          (!(sample >= curBucket) &&
          (sample >= prevBucket) &&
          (dist_val > zero));

      if (inBucket) {
        atomicMax(&foundPos, cat);
        found = true;
      }

      prevHighProb = prevHighProb + smem[blockDim.x - 1];
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      if (found) {
        dest[curDist] = foundPos;
      } else {
        for (int cat = categories - 1; cat >= 0; --cat) {
          if (dist[curDist * stride_dist + cat * stride_categories] > zero) {
            dest[curDist] = cat;
            break;
          }
        }
      }
    }
  }
}
