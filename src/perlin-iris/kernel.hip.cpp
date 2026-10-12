#include <hip/hip_runtime.h>
#include <cstdint>

static constexpr uint64_t WIN_WIDTH = 61440;
static constexpr uint64_t WIN_HEIGHT = 34560;
static constexpr float INVSQRT2 = .70710678118654752440f;
static constexpr int N_GRADIENTS = 8;

struct NoiseParams {
  float ppu;
  int seed;
  int octaves;
  float lacunarity;
  float persistence;
};

__device__ const int kHash[256] = {
  151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
  140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
  247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
  57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
  74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
  60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
  65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
  200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
  52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
  207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
  119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
  129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
  218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
  81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
  184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
  222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180
};

__device__ const float kGradientX[N_GRADIENTS] = {
  1.f, -1.f, 0.f, 0.f, INVSQRT2, -INVSQRT2, INVSQRT2, -INVSQRT2
};

__device__ const float kGradientY[N_GRADIENTS] = {
  0.f, 0.f, 1.f, -1.f, INVSQRT2, INVSQRT2, -INVSQRT2, -INVSQRT2
};

__device__ float lerp(float a, float b, float t)
{
  return a + t * (b - a);
}

__host__ __device__ float dot(float2 a, float2 b)
{
  return a.x * b.x + a.y * b.y;
}

__device__ float smooth(float t)
{
  return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

__device__ float noise_at(float x, float y, int seed)
{
  const int ix = static_cast<int>(x);
  const int iy = static_cast<int>(y);
  const float wx = x - ix;
  const float wy = y - iy;

  const int ix0 = ix & 255;
  const int iy0 = iy & 255;
  const int ix1 = (ix0 + 1) & 255;
  const int iy1 = (iy0 + 1) & 255;
  const int h0 = kHash[ix0];
  const int h1 = kHash[ix1];
  const int i_tl = (kHash[(h0 + iy0) & 255] + seed) % N_GRADIENTS;
  const int i_tr = (kHash[(h1 + iy0) & 255] + seed) % N_GRADIENTS;
  const int i_bl = (kHash[(h0 + iy1) & 255] + seed) % N_GRADIENTS;
  const int i_br = (kHash[(h1 + iy1) & 255] + seed) % N_GRADIENTS;

  const float2 g_top_left = make_float2(kGradientX[i_tl], kGradientY[i_tl]);
  const float2 g_top_right = make_float2(kGradientX[i_tr], kGradientY[i_tr]);
  const float2 g_bot_left = make_float2(kGradientX[i_bl], kGradientY[i_bl]);
  const float2 g_bot_right = make_float2(kGradientX[i_br], kGradientY[i_br]);

  const float d_top_left = dot(g_top_left, make_float2(wx, wy));
  const float d_top_right = dot(g_top_right, make_float2(wx - 1.f, wy));
  const float d_bot_left = dot(g_bot_left, make_float2(wx, wy - 1.f));
  const float d_bot_right = dot(g_bot_right, make_float2(wx - 1.f, wy - 1.f));

  const float tx = smooth(wx);
  const float ty = smooth(wy);
  const float left_interp = lerp(d_top_left, d_bot_left, ty);
  const float right_interp = lerp(d_top_right, d_bot_right, ty);
  return (lerp(left_interp, right_interp, tx) + 1.0f) * 0.5f;
}

__device__ float sum_octaves(float x, float y, NoiseParams params)
{
  float frequency = 1.f;
  float sum = noise_at(x * frequency, y * frequency, params.seed);
  float amplitude = 1.f;
  float range = 1.f;
  for (int i = 1; i < params.octaves; i++) {
    frequency *= params.lacunarity;
    amplitude *= params.persistence;
    range += amplitude;
    sum += amplitude * noise_at(x * frequency, y * frequency, params.seed);
  }
  return sum / range;
}

extern "C" __global__ void perlin(int y_start, int height, NoiseParams params, uint8_t* out_pixels)
{
  const uint64_t px = blockDim.x * blockIdx.x + threadIdx.x;
  const uint64_t py = blockDim.y * blockIdx.y + threadIdx.y + static_cast<uint64_t>(y_start);

  if (px >= WIN_WIDTH || py >= static_cast<uint64_t>(y_start + height) || py >= WIN_HEIGHT) return;

  const float noise = sum_octaves(px / params.ppu, py / params.ppu, params);
  const uint64_t base_idx = 4 * (py * WIN_WIDTH + px);
  const uint8_t val = static_cast<uint8_t>(noise * 255.f);

  out_pixels[base_idx + 0] = val;
  out_pixels[base_idx + 1] = val;
  out_pixels[base_idx + 2] = val;
  out_pixels[base_idx + 3] = 255;
}
