#include <hip/hip_runtime.h>

static constexpr float R00 = -0.9111348390579224f;
static constexpr float R01 =  0.0751304179430008f;
static constexpr float R02 = -0.4052018225193024f;
static constexpr float R10 = -0.3360927104949951f;
static constexpr float R11 = -0.7044632434844971f;
static constexpr float R12 =  0.6251187324523926f;
static constexpr float R20 = -0.2384843230247498f;
static constexpr float R21 =  0.7057529091835022f;
static constexpr float R22 =  0.6671117544174194f;
static constexpr float T0 = 0.1f, T1 = -0.2f, T2 = 0.3f;
static constexpr float D0 = 0.03f, D1 = -0.15f, D2 = 0.001f, D3 = 0.001f, D4 = 0.05f;
static constexpr float FX = 1200.0f, CX = 400.0f, FY = 1200.0f, CY = 300.0f;
static constexpr int POINT_STEP = 8;

__device__ inline void project_point(float p0, float p1, float p2, float& px, float& py, float& pz, int& valid)
{
  float pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02;
  float pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12;
  float pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22;
  bool close = pt2 > 2.5f;
  float denom = close ? pt2 : 1.0f;
  float tmpx = pt0 / denom;
  float tmpy = pt1 / denom;
  float r2 = tmpx * tmpx + tmpy * tmpy;
  float tmpdist = 1.0f + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2;
  float ix = tmpx * tmpdist + 2.0f * D2 * tmpx * tmpy + D3 * (r2 + 2.0f * tmpx * tmpx);
  float iy = tmpy * tmpdist + D2 * (r2 + 2.0f * tmpy * tmpy) + 2.0f * D3 * tmpx * tmpy;
  float ux = FX * ix + CX;
  float uy = FY * iy + CY;
  if (close) {
    px = ux + 0.5f;
    py = uy + 0.5f;
    pz = pt2 * 100.0f;
    valid = 1;
  } else {
    px = 0.0f;
    py = 0.0f;
    pz = 0.0f;
    valid = 0;
  }
}

extern "C" __global__ void project_kernel(
    const float* __restrict__ cp,
    float* __restrict__ out_x,
    float* __restrict__ out_y,
    float* __restrict__ out_z,
    int* __restrict__ out_valid,
    int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* pt = cp + i * POINT_STEP;
  float px, py, pz;
  int v;
  project_point(pt[0], pt[1], pt[2], px, py, pz, v);
  out_x[i] = px;
  out_y[i] = py;
  out_z[i] = pz;
  out_valid[i] = v;
}
