#include <iris/iris.hpp>

#include "sobel.h"
#include "SDKBitMap.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool compare(const float* refData, const float* data, int length, float epsilon = 1e-6f)
{
  float error = 0.0f;
  float ref = 0.0f;
  for (int i = 1; i < length; ++i) {
    float diff = refData[i] - data[i];
    error += diff * diff;
    ref += refData[i] * refData[i];
  }
  float normRef = sqrtf(ref);
  if (fabs(ref) < 1e-7f) return false;
  float normError = sqrtf(error);
  error = normError / normRef;
  return error < epsilon;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <path to file> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const char* filePath = argv[1];
  const int iterations = std::atoi(argv[2]);
  SDKBitMap inputBitmap;
  inputBitmap.load(filePath);
  if (!inputBitmap.isLoaded()) {
    std::printf("Failed to load input image!");
    platform.finalize();
    return SDK_FAILURE;
  }

  const int height = inputBitmap.getHeight();
  const int width = inputBitmap.getWidth();
  const int pixelSize = sizeof(uchar4);
  const int imageSize = width * height * pixelSize;
  std::printf("Image height = %d and width = %d\n", height, width);

  uchar4* inputImageData = static_cast<uchar4*>(std::malloc(imageSize));
  uchar4* outputImageData = static_cast<uchar4*>(std::malloc(imageSize));
  uchar4* verificationOutput = static_cast<uchar4*>(std::malloc(imageSize));
  if (!inputImageData || !outputImageData || !verificationOutput) {
    std::fprintf(stderr, "host allocation failed\n");
    return 1;
  }
  std::memset(outputImageData, 0, imageSize);
  std::memset(verificationOutput, 0, imageSize);
  uchar4* pixelData = inputBitmap.getPixels();
  if (pixelData == nullptr) std::printf("Failed to read pixel Data!");
  std::memcpy(inputImageData, pixelData, imageSize);

  iris::DMem mem_input(inputImageData, imageSize);
  iris::DMem mem_output(outputImageData, imageSize);
  const int blockSizeX = 16;
  const int blockSizeY = 16;
  size_t gws[2] = {static_cast<size_t>((width / blockSizeX) * blockSizeX),
                   static_cast<size_t>((height / blockSizeY) * blockSizeY)};
  size_t lws[2] = {static_cast<size_t>(blockSizeX), static_cast<size_t>(blockSizeY)};
  uint uwidth = static_cast<uint>(width);
  uint uheight = static_cast<uint>(height);

  std::printf("Executing kernel for %d iterations", iterations);
  std::printf("-------------------------------------------\n");
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    iris::Task task;
    void* params[4] = {&mem_input, &mem_output, &uwidth, &uheight};
    int pinfo[4] = {iris_r, iris_w, sizeof(uwidth), sizeof(uheight)};
    task.kernel("sobel_filter", 2, nullptr, gws, lws, 4, params, pinfo);
    if (i == iterations - 1) task.flush_out(mem_output);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / iterations);

  reference(verificationOutput, inputImageData, width, height, pixelSize);

  float* outputDevice = static_cast<float*>(std::malloc(imageSize * sizeof(float)));
  float* outputReference = static_cast<float*>(std::malloc(imageSize * sizeof(float)));
  for (int i = 0; i < width * height; i++) {
    outputDevice[i * 4 + 0] = outputImageData[i].x;
    outputDevice[i * 4 + 1] = outputImageData[i].y;
    outputDevice[i * 4 + 2] = outputImageData[i].z;
    outputDevice[i * 4 + 3] = outputImageData[i].w;
    outputReference[i * 4 + 0] = verificationOutput[i].x;
    outputReference[i * 4 + 1] = verificationOutput[i].y;
    outputReference[i * 4 + 2] = verificationOutput[i].z;
    outputReference[i * 4 + 3] = verificationOutput[i].w;
  }

  std::printf("%s\n", compare(outputReference, outputDevice, imageSize) ? "PASS" : "FAIL");

  std::free(outputDevice);
  std::free(outputReference);
  std::free(verificationOutput);
  std::free(inputImageData);
  std::free(outputImageData);
  platform.finalize();
  return SDK_SUCCESS;
}
