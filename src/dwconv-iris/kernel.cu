#include <cuda_runtime.h>

extern "C" __global__ void conv_depthwise2d_forward_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    bool biasEnabled,
    int totalElements,
    int outputChannels,
    int depthwiseMultiplier,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    int kernelWidth,
    int kernelHeight,
    int strideWidth,
    int strideHeight,
    int padWidth,
    int padHeight)
{
  int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < totalElements) {
    int indtmp1 = linearIndex / outputWidth;
    const int w = linearIndex - indtmp1 * outputWidth;
    int indtmp2 = indtmp1 / outputHeight;
    const int h = indtmp1 - indtmp2 * outputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / outputChannels;
    const int c = indtmp1 - indtmp2 * outputChannels;
    const int n = indtmp2;

    int inputChannel = c;
    int inputChannels = outputChannels;
    if (depthwiseMultiplier != 1) {
      inputChannel /= depthwiseMultiplier;
      inputChannels /= depthwiseMultiplier;
    }

    int weightOffset = c * kernelHeight * kernelWidth;
    float value = biasEnabled ? bias[c] : 0.0f;
    const int offset0 =
        (n * inputChannels + inputChannel) * inputHeight * inputWidth;

    for (int kH = 0; kH < kernelHeight; ++kH) {
      for (int kW = 0; kW < kernelWidth; ++kW) {
        const int h_in = -padHeight + h * strideHeight + kH;
        const int w_in = -padWidth + w * strideWidth + kW;

        if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) &&
            (w_in < inputWidth)) {
          const int offset = offset0 + h_in * inputWidth + w_in;
          value += weight[weightOffset] * input[offset];
        }
        ++weightOffset;
      }
    }
    output[linearIndex] = value;
  }
}
