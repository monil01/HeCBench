#pragma once

#include <cstdint>

struct float3 {
  float x;
  float y;
  float z;
};

struct float4 {
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

inline float4 make_float4(float x, float y, float z, float w)
{
  return {x, y, z, w};
}

template <typename T>
void cpuDetectionOverlayBox(const T* __restrict input,
                            T* __restrict output,
                            int imgWidth,
                            int imgHeight,
                            int x0,
                            int y0,
                            int boxWidth,
                            int boxHeight,
                            const float4 color)
{
  for (int box_y = 0; box_y < boxHeight; box_y++) {
    for (int box_x = 0; box_x < boxWidth; box_x++) {
      const int x = box_x + x0;
      const int y = box_y + y0;
      if (x < imgWidth && y < imgHeight) {
        T px = input[y * imgWidth + x];
        const float alpha = color.w / 255.0f;
        const float ialph = 1.0f - alpha;
        px.x = alpha * color.x + ialph * px.x;
        px.y = alpha * color.y + ialph * px.y;
        px.z = alpha * color.z + ialph * px.z;
        output[y * imgWidth + x] = px;
      }
    }
  }
}

template <typename T>
int reference(T* input, T* output, uint32_t width, uint32_t height,
              Box* detections, int numDetections, float4 colors)
{
  if (!input || !output || width == 0 || height == 0 || !detections || numDetections == 0) {
    return 1;
  }
  for (int n = 0; n < numDetections; n++) {
    cpuDetectionOverlayBox<T>(input, output, width, height, detections[n].left,
                              detections[n].top, detections[n].width,
                              detections[n].height, colors);
  }
  return 0;
}
