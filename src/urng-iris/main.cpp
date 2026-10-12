#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "SDKBitMap.h"
#include "urng.h"

int main(int argc, char** argv)
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
    std::cout << "Failed to load input image!";
    platform.finalize();
    return -1;
  }

  const int height = inputBitmap.getHeight();
  const int width = inputBitmap.getWidth();
  const int size = height * width;
  const size_t imageSize = size * sizeof(uchar4);

  std::cout << "Image " << filePath;
  std::cout << " height: " << height;
  std::cout << " width: " << width << std::endl;

  uchar4* inputImageData = static_cast<uchar4*>(std::malloc(imageSize));
  uchar4* outputImageData = static_cast<uchar4*>(std::malloc(imageSize));
  if (!inputImageData || !outputImageData) {
    std::cerr << "allocation failure\n";
    platform.finalize();
    return 1;
  }
  std::memset(outputImageData, 0, imageSize);

  uchar4* pixelData = inputBitmap.getPixels();
  if (pixelData == nullptr) {
    std::cout << "Failed to read pixel Data!";
    std::free(inputImageData);
    std::free(outputImageData);
    platform.finalize();
    return -1;
  }
  std::memcpy(inputImageData, pixelData, imageSize);

  const int factor = FACTOR;
  iris::DMem inputImageBuffer(inputImageData, imageSize);
  iris::DMem outputImageBuffer(outputImageData, imageSize);

  const size_t gws[1] = {static_cast<size_t>(((size + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE)};
  const size_t lws[1] = {GROUP_SIZE};

  std::cout << "Executing kernel for " << iterations << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    iris::Task task;
    void* params[4] = {&inputImageBuffer, &outputImageBuffer, const_cast<int*>(&size), const_cast<int*>(&factor)};
    int pinfo[4] = {iris_r, iris_w, sizeof(size), sizeof(factor)};
    task.kernel("noise_uniform", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
    if (i + 1 == iterations) task.flush_out(outputImageBuffer);
    task.submit(iris_gpu, nullptr, 1);
  }
  const auto end = std::chrono::steady_clock::now();
  const double time_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3;
  std::cout << "Average kernel execution time: " << time_us / iterations << " (us)\n";

  float mean = 0.0f;
  for (int i = 0; i < size; i++) {
    mean += outputImageData[i].x - inputImageData[i].x;
    mean += outputImageData[i].y - inputImageData[i].y;
    mean += outputImageData[i].z - inputImageData[i].z;
    mean += outputImageData[i].w - inputImageData[i].w;
  }
  mean /= (imageSize * factor);
  std::cout << "The averaged mean of the image: " << mean << std::endl;

  if (std::fabs(mean) < 1.0f) {
    std::cout << "PASS\n" << std::endl;
  } else {
    std::cout << "FAIL\n" << std::endl;
  }

  std::free(inputImageData);
  std::free(outputImageData);
  platform.finalize();
  return 0;
}
