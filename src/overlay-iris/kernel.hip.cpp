#include <hip/hip_runtime.h>

struct Pixel3 {
  float x;
  float y;
  float z;
};

struct Color4 {
  float x;
  float y;
  float z;
  float w;
};

struct Box {
  int width;
  int height;
  int left;
  int top;
};

extern "C" __global__ void detection_overlay_kernel(const Pixel3* __restrict__ input,
                                                     Pixel3* __restrict__ output,
                                                     int imgWidth,
                                                     int imgHeight,
                                                     const Box* __restrict__ detections,
                                                     int numDetections,
                                                     Color4 color)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= numDetections) {
    return;
  }

  const Box box = detections[n];
  const float alpha = color.w / 255.0f;
  const float ialph = 1.0f - alpha;
  for (int box_y = 0; box_y < box.height; box_y++) {
    const int y = box_y + box.top;
    if (y >= imgHeight) {
      continue;
    }
    for (int box_x = 0; box_x < box.width; box_x++) {
      const int x = box_x + box.left;
      if (x >= imgWidth) {
        continue;
      }
      Pixel3 px = input[y * imgWidth + x];
      px.x = alpha * color.x + ialph * px.x;
      px.y = alpha * color.y + ialph * px.y;
      px.z = alpha * color.z + ialph * px.z;
      output[y * imgWidth + x] = px;
    }
  }
}
