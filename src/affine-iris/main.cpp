#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "reference.h"

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::printf("Usage: %s <input image> <output image> <iterations>\n", argv[0]);
    return -1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  auto* input_image = static_cast<unsigned short*>(std::aligned_alloc(1024, sizeof(unsigned short) * Y_SIZE * X_SIZE));
  auto* output_image = static_cast<unsigned short*>(std::aligned_alloc(1024, sizeof(unsigned short) * Y_SIZE * X_SIZE));
  auto* output_image_ref = static_cast<unsigned short*>(std::aligned_alloc(1024, sizeof(unsigned short) * Y_SIZE * X_SIZE));
  if (!input_image || !output_image || !output_image_ref) {
    std::fprintf(stderr, "allocation failure\n");
    platform.finalize();
    return 1;
  }

  std::cout << "Reading input image...\n";
  const char* inputImageFilename = argv[1];
  FILE* input_file = std::fopen(inputImageFilename, "rb");
  if (!input_file) {
    std::printf("Error: Unable to open input image file %s!\n", inputImageFilename);
    platform.finalize();
    return 1;
  }

  std::printf("\n");
  std::printf("   Reading RAW Image\n");
  const size_t image_size_bytes = sizeof(unsigned short) * X_SIZE * Y_SIZE;
  size_t items_read = std::fread(input_image, image_size_bytes, 1, input_file);
  std::printf("   Bytes read = %d\n\n", (int)(items_read * image_size_bytes));
  std::fclose(input_file);

  const int iterations = std::atoi(argv[3]);
  std::memset(output_image, 0, image_size_bytes);

  iris::DMem input_mem(input_image, image_size_bytes);
  iris::DMem output_mem(output_image, image_size_bytes);

  const size_t gws[2] = {X_SIZE, Y_SIZE};
  const size_t lws[2] = {16, 16};

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    iris::Task task;
    void* params[2] = {&input_mem, &output_mem};
    int pinfo[2] = {iris_r, iris_w};
    task.kernel("affine", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 2, params, pinfo);
    if (i + 1 == iterations) task.flush_out(output_mem);
    task.submit(iris_gpu, nullptr, 1);
  }
  const auto end = std::chrono::steady_clock::now();
  const double time_s = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;
  std::cout << "   Average kernel execution time " << time_s / iterations << " (s)\n";

  affine_reference(input_image, output_image_ref);
  int max_error = 0;
  for (int y = 0; y < Y_SIZE; y++) {
    for (int x = 0; x < X_SIZE; x++) {
      max_error = std::max(max_error, std::abs(output_image[y * X_SIZE + x] - output_image_ref[y * X_SIZE + x]));
    }
  }
  std::printf("   Max output error is %d\n\n", max_error);

  std::printf("   Writing RAW Image\n");
  const char* outputImageFilename = argv[2];
  FILE* output_file = std::fopen(outputImageFilename, "wb");
  if (!output_file) {
    std::printf("Error: Unable to write  image file %s!\n", outputImageFilename);
    platform.finalize();
    return 1;
  }
  size_t items_written = std::fwrite(output_image, image_size_bytes, 1, output_file);
  std::printf("   Bytes written = %d\n\n", (int)(items_written * image_size_bytes));
  std::fclose(output_file);

  std::free(input_image);
  std::free(output_image);
  std::free(output_image_ref);
  platform.finalize();
  return 0;
}
