#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "reference.h"

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <width> <height>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int width = std::atoi(argv[1]);
  const int height = std::atoi(argv[2]);
  const int img_size = width * height;
  const size_t img_size_byte = sizeof(float3) * static_cast<size_t>(img_size);

  std::srand(123);
  float3* input = static_cast<float3*>(std::malloc(img_size_byte));
  float3* output = static_cast<float3*>(std::malloc(img_size_byte));
  float3* ref_output = static_cast<float3*>(std::malloc(img_size_byte));
  if (!input || !output || !ref_output) {
    std::fprintf(stderr, "image allocation failed\n");
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < img_size; i++) {
    ref_output[i].x = output[i].x = input[i].x = std::rand() % 256;
    ref_output[i].y = output[i].y = input[i].y = std::rand() % 256;
    ref_output[i].z = output[i].z = input[i].z = std::rand() % 256;
  }

  const int numDetections = static_cast<int>(img_size * 0.8f);
  Box* detections = static_cast<Box*>(std::malloc(sizeof(Box) * static_cast<size_t>(numDetections)));
  if (!detections) {
    std::fprintf(stderr, "detection allocation failed\n");
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < numDetections; i++) {
    detections[i].width = 64 + std::rand() % 128;
    detections[i].height = 64 + std::rand() % 128;
    detections[i].left = std::rand() % (width - 64);
    detections[i].top = std::rand() % (height - 64);
  }

  float4 colors = make_float4(255, 204, 203, 1);
  iris::DMem mem_input(input, img_size_byte);
  iris::DMem mem_output(output, img_size_byte);
  iris::DMem mem_detections(detections, sizeof(Box) * static_cast<size_t>(numDetections));

  const size_t lws[1] = {256};
  const size_t gws[1] = {static_cast<size_t>((numDetections + 255) / 256 * 256)};
  int width_arg = width;
  int height_arg = height;
  int num_arg = numDetections;

  auto start = std::chrono::steady_clock::now();
  iris::Task task;
  void* params[7] = {&mem_input, &mem_output, &width_arg, &height_arg,
                     &mem_detections, &num_arg, &colors};
  int pinfo[7] = {iris_r, iris_rw, sizeof(width_arg), sizeof(height_arg),
                  iris_r, sizeof(num_arg), sizeof(colors)};
  task.kernel("detection_overlay_kernel", 1, nullptr, const_cast<size_t*>(gws),
              const_cast<size_t*>(lws), 7, params, pinfo);
  task.flush_out(mem_output);
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  mem_output.fetch(output);
  reference<float3>(input, ref_output, width, height, detections, numDetections, colors);

  bool ok = true;
  for (int i = 0; i < img_size; i++) {
    if ((std::fabs(ref_output[i].x - output[i].x) > 1e-3f) ||
        (std::fabs(ref_output[i].y - output[i].y) > 1e-3f) ||
        (std::fabs(ref_output[i].z - output[i].z) > 1e-3f)) {
      std::printf("Error at index %d\n", i);
      ok = false;
      break;
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  std::free(input);
  std::free(output);
  std::free(ref_output);
  std::free(detections);
  platform.finalize();
  return ok ? 0 : 1;
}
