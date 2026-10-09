#include <hip/hip_runtime.h>

struct storeElement {
  int tag;
  int metric;
  unsigned long long int time;
  float value;
};

struct kurtosisResult {
  int count;
  float mean;
  float m2;
  float m3;
  float m4;
};

__device__ kurtosisResult combine_kurtosis_device(const kurtosisResult& x, const kurtosisResult& y)
{
  if (x.count == 0) {
    return y;
  }
  if (y.count == 0) {
    return x;
  }
  const float count = static_cast<float>(x.count + y.count);
  const float count2 = count * count;
  const float count3 = count2 * count;
  const float delta = y.mean - x.mean;
  const float delta2 = delta * delta;
  const float delta3 = delta2 * delta;
  const float delta4 = delta3 * delta;

  kurtosisResult result;
  result.count = x.count + y.count;
  result.mean = x.mean + delta * y.count / count;
  result.m2 = x.m2 + y.m2 + delta2 * x.count * y.count / count;
  result.m3 = x.m3 + y.m3;
  result.m3 += delta3 * x.count * y.count * (x.count - y.count) / count2;
  result.m3 += 3.0f * delta * (x.count * y.m2 - y.count * x.m2) / count;
  result.m4 = x.m4 + y.m4;
  result.m4 += delta4 * x.count * y.count *
               (x.count * x.count - x.count * y.count + y.count * y.count) / count3;
  result.m4 += 6.0f * delta2 *
               (x.count * x.count * y.m2 + y.count * y.count * x.m2) / count2;
  result.m4 += 4.0f * delta * (x.count * y.m3 - y.count * x.m3) / count;
  return result;
}

extern "C" __global__ void kurtosis_partials(const storeElement* elements,
                                              kurtosisResult* partials,
                                              int elemCount)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ kurtosisResult shared[256];
  kurtosisResult result{0, 0.0f, 0.0f, 0.0f, 0.0f};
  if (idx < elemCount) {
    result.count = 1;
    result.mean = elements[idx].value;
  }
  shared[threadIdx.x] = result;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = combine_kurtosis_device(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

extern "C" __global__ void kurtosis_reduce(const kurtosisResult* input,
                                            kurtosisResult* output,
                                            int count)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ kurtosisResult shared[256];
  kurtosisResult result{0, 0.0f, 0.0f, 0.0f, 0.0f};
  if (idx < count) {
    result = input[idx];
  }
  shared[threadIdx.x] = result;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = combine_kurtosis_device(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[blockIdx.x] = shared[0];
  }
}
