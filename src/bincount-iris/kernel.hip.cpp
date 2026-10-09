#include <hip/hip_runtime.h>

#define threadsPerBlock 256

extern "C" __global__ void zero_int(int* output, int output_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < output_size) output[i] = 0;
}

__device__ static int get_bin(float v, float minvalue, float maxvalue,
                              int nbins) {
  int bin = (v - minvalue) * nbins / (maxvalue - minvalue);
  if (bin == nbins) bin--;
  return bin;
}

extern "C" __global__ void bincount_global_int_float_int(
    int* output, const float* input, int nbins, float minvalue, float maxvalue,
    int input_size, int) {
  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < input_size; linearIndex += gridDim.x * blockDim.x) {
    const float v = input[linearIndex];
    if (v >= minvalue && v <= maxvalue) {
      const int bin = get_bin(v, minvalue, maxvalue, nbins);
      atomicAdd(&output[bin], 1);
    }
  }
}
