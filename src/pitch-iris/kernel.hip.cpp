#include <hip/hip_runtime.h>

__device__ float sigmoid(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

extern "C" __global__ void parallelPitched2DAccess(
    float* devPtr,
    size_t pitch,
    int width,
    int height)
{
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < height && c < width) {
    float* row = reinterpret_cast<float*>(reinterpret_cast<char*>(devPtr) + r * pitch);
    row[c] = sigmoid(row[c]);
  }
}

extern "C" __global__ void parallelSimple2DAccess(float* elem, int width, int height)
{
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < height && c < width) {
    elem[r * width + c] = sigmoid(elem[r * width + c]);
  }
}

extern "C" __global__ void parallelPitched3DAccess(
    float* devPtr,
    size_t pitch,
    int width,
    int height,
    int depth)
{
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (z < depth && y < height && x < width) {
    const size_t slicePitch = pitch * height;
    char* slice = reinterpret_cast<char*>(devPtr) + static_cast<size_t>(z) * slicePitch;
    float* row = reinterpret_cast<float*>(slice + static_cast<size_t>(y) * pitch);
    row[x] = sigmoid(row[x]);
  }
}

extern "C" __global__ void parallelSimple3DAccess(
    float* elem,
    int width,
    int height,
    int depth)
{
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (z < depth && y < height && x < width) {
    const size_t idx = (static_cast<size_t>(z) * height + y) * width + x;
    elem[idx] = sigmoid(elem[idx]);
  }
}
