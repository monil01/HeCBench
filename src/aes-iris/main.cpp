#include <iris/iris.hpp>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "../include/SDKBitMap.h"
#include "../aes-cuda/aes.h"

__host__ __device__ uchar galoisMultiplication(uchar a, uchar b) {
  uchar p = 0;
  for (unsigned int i = 0; i < 8; ++i) {
    if ((b & 1) == 1) p ^= a;
    uchar hiBitSet = (a & 0x80);
    a <<= 1;
    if (hiBitSet == 0x80) a ^= 0x1b;
    b >>= 1;
  }
  return p;
}

#include "../aes-cuda/reference.cu"
#include "../aes-cuda/utils.cu"

int main(int argc, char* argv[]) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  if (argc != 4) {
    std::printf("Usage: %s <iterations> <0 or 1> <path to bitmap image file>\n", argv[0]);
    std::printf("0=encrypt, 1=decrypt\n");
    platform.finalize();
    return 1;
  }

  const unsigned int keySizeBits = 128;
  const unsigned int rounds = 10;
  const unsigned int seed = 123;

  const int iterations = std::atoi(argv[1]);
  const bool decrypt = std::atoi(argv[2]);
  const char* filePath = argv[3];

  SDKBitMap image;
  image.load(filePath);
  const int width = image.getWidth();
  const int height = image.getHeight();
  if (width <= 0 || height <= 0) {
    platform.finalize();
    return 1;
  }

  std::cout << "Image width and height: " << width << " " << height << std::endl;

  uchar4* pixels = image.getPixels();
  unsigned int sizeBytes = width * height * sizeof(uchar);
  uchar* input = static_cast<uchar*>(std::malloc(sizeBytes));

  if (decrypt) {
    convertGrayToGray(pixels, input, height, width);
  } else {
    convertColorToGray(pixels, input, height, width);
  }

  unsigned int keySize = keySizeBits / 8;
  unsigned int keySizeBytes = keySize * sizeof(uchar);
  uchar* key = static_cast<uchar*>(std::malloc(keySizeBytes));
  fillRandom<uchar>(key, keySize, 1, 0, 255, seed);

  unsigned int explandedKeySize = (rounds + 1) * keySize;
  unsigned int explandedKeySizeBytes = explandedKeySize * sizeof(uchar);
  uchar* expandedKey = static_cast<uchar*>(std::malloc(explandedKeySizeBytes));
  uchar* roundKey = static_cast<uchar*>(std::malloc(explandedKeySizeBytes));

  keyExpansion(key, expandedKey, keySize, explandedKeySize);
  for (unsigned int i = 0; i < rounds + 1; ++i) {
    createRoundKey(expandedKey + keySize * i, roundKey + keySize * i);
  }

  uchar* output = static_cast<uchar*>(std::malloc(sizeBytes));

  iris::DMem mem_input(input, sizeBytes);
  iris::DMem mem_output(output, sizeBytes);
  iris::DMem mem_roundKey(roundKey, explandedKeySizeBytes);
  iris::DMem mem_sbox(sbox, 256);
  iris::DMem mem_rsbox(rsbox, 256);

  std::cout << "Executing kernel for " << iterations << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  size_t gws[2] = {static_cast<size_t>(width / 4), static_cast<size_t>(height)};
  size_t lws[2] = {1, 4};
  unsigned int kernel_width = static_cast<unsigned int>(width);
  unsigned int kernel_rounds = rounds;

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    iris::Task task;
    void* params[] = {
        &mem_output,
        &mem_input,
        &mem_roundKey,
        decrypt ? static_cast<void*>(&mem_rsbox) : static_cast<void*>(&mem_sbox),
        &kernel_width,
        &kernel_rounds};
    int pinfo[] = {
        iris_w,
        iris_r,
        iris_r,
        iris_r,
        sizeof(kernel_width),
        sizeof(kernel_rounds)};
    task.kernel(decrypt ? "AESDecrypt" : "AESEncrypt", 2, nullptr, gws, lws, 6, params, pinfo);
    if (i == iterations - 1) task.flush_out(mem_output);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / iterations << " (s)\n";

  uchar* verificationOutput = static_cast<uchar*>(std::malloc(sizeBytes));
  reference(verificationOutput, input, roundKey, explandedKeySize, width, height,
            decrypt, rounds, keySize);

  if (std::memcmp(output, verificationOutput, sizeBytes) == 0) {
    std::cout << "Pass\n";
  } else {
    std::cout << "Fail\n";
  }

  std::free(input);
  std::free(key);
  std::free(expandedKey);
  std::free(roundKey);
  std::free(output);
  std::free(verificationOutput);

  platform.finalize();
  return 0;
}
