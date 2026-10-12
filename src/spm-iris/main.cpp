#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct int3 {
  int x;
  int y;
  int z;
};

#define NUM_THREADS 128
#define NUM_BLOCKS 256

static float interp(const int3 d, const unsigned char f[], float x, float y, float z)
{
  int ix, iy, iz;
  float dx1, dy1, dz1, dx2, dy2, dz2;
  int k111, k112, k121, k122, k211, k212, k221, k222;
  float vf;
  const unsigned char* ff;

  ix = floorf(x); dx1 = x - ix; dx2 = 1.f - dx1;
  iy = floorf(y); dy1 = y - iy; dy2 = 1.f - dy1;
  iz = floorf(z); dz1 = z - iz; dz2 = 1.f - dz1;

  ff = f + ix - 1 + d.x * (iy - 1 + d.y * (iz - 1));
  k222 = ff[0]; k122 = ff[1];
  k212 = ff[d.x]; k112 = ff[d.x + 1];
  ff += d.x * d.y;
  k221 = ff[0]; k121 = ff[1];
  k211 = ff[d.x]; k111 = ff[d.x + 1];

  vf = (((k222 * dx2 + k122 * dx1) * dy2 + (k212 * dx2 + k112 * dx1) * dy1)) * dz2 +
       (((k221 * dx2 + k121 * dx1) * dy2 + (k211 * dx2 + k111 * dx1) * dy1)) * dz1;

  return vf;
}

static void spm_reference(const float* M, const int data_size,
                          const unsigned char* g_d,
                          const unsigned char* f_d,
                          const int3 dg, const int3 df,
                          unsigned char* ivf_d,
                          unsigned char* ivg_d,
                          bool* data_threshold_d)
{
  const float ran[] = {
    0.656619,0.891183,0.488144,0.992646,0.373326,0.531378,0.181316,0.501944,0.422195,
    0.660427,0.673653,0.95733,0.191866,0.111216,0.565054,0.969166,0.0237439,0.870216,
    0.0268766,0.519529,0.192291,0.715689,0.250673,0.933865,0.137189,0.521622,0.895202,
    0.942387,0.335083,0.437364,0.471156,0.14931,0.135864,0.532498,0.725789,0.398703,
    0.358419,0.285279,0.868635,0.626413,0.241172,0.978082,0.640501,0.229849,0.681335,
    0.665823,0.134718,0.0224933,0.262199,0.116515,0.0693182,0.85293,0.180331,0.0324186,
    0.733926,0.536517,0.27603,0.368458,0.0128863,0.889206,0.866021,0.254247,0.569481,
    0.159265,0.594364,0.3311,0.658613,0.863634,0.567623,0.980481,0.791832,0.152594,
    0.833027,0.191863,0.638987,0.669,0.772088,0.379818,0.441585,0.48306,0.608106,
    0.175996,0.00202556,0.790224,0.513609,0.213229,0.10345,0.157337,0.407515,0.407757,
    0.0526927,0.941815,0.149972,0.384374,0.311059,0.168534,0.896648};

  int x_datasize = dg.x - 2;
  int y_datasize = dg.y - 2;

  for (int i = 0; i < data_size; i++) {
    float xx_temp = (i % x_datasize) + 1.f;
    float yy_temp = (static_cast<int>(floorf(static_cast<float>(i) / x_datasize)) % y_datasize) + 1.f;
    float zz_temp = (floorf(static_cast<float>(i) / x_datasize)) / y_datasize + 1.f;

    float rx = xx_temp + ran[i % 97];
    float ry = yy_temp + ran[i % 97];
    float rz = zz_temp + ran[i % 97];

    float xp = M[0] * rx + M[4] * ry + M[8] * rz + M[12];
    float yp = M[1] * rx + M[5] * ry + M[9] * rz + M[13];
    float zp = M[2] * rx + M[6] * ry + M[10] * rz + M[14];

    if (zp >= 1.f && zp < df.z && yp >= 1.f && yp < df.y && xp >= 1.f && xp < df.x) {
      ivf_d[i] = floorf(interp(df, f_d, xp, yp, zp) + 0.5f);
      ivg_d[i] = floorf(interp(dg, g_d, rx, ry, rz) + 0.5f);
      data_threshold_d[i] = true;
    } else {
      ivf_d[i] = 0;
      ivg_d[i] = 0;
      data_threshold_d[i] = false;
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <dimension> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int v = std::atoi(argv[1]);
  int repeat = std::atoi(argv[2]);

  int3 g_vol = {v, v, v};
  int3 f_vol = {v, v, v};

  const int data_size = (g_vol.x + 1) * (g_vol.y + 1) * (g_vol.z + 5);
  const int vol_size = g_vol.x * g_vol.y * g_vol.z;

  int* hist_d = static_cast<int*>(std::malloc(65536 * sizeof(int)));
  int* hist_h = static_cast<int*>(std::malloc(65536 * sizeof(int)));
  std::memset(hist_d, 0, sizeof(int) * 65536);
  std::memset(hist_h, 0, sizeof(int) * 65536);

  unsigned char* ivf_h = static_cast<unsigned char*>(std::malloc(vol_size * sizeof(unsigned char)));
  unsigned char* ivg_h = static_cast<unsigned char*>(std::malloc(vol_size * sizeof(unsigned char)));
  bool* data_threshold_h = static_cast<bool*>(std::malloc(vol_size * sizeof(bool)));

  std::srand(123);
  float M_h[16];
  for (int i = 0; i < 16; i++) M_h[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

  unsigned char* g_h = static_cast<unsigned char*>(std::malloc(data_size * sizeof(unsigned char)));
  unsigned char* f_h = static_cast<unsigned char*>(std::malloc(data_size * sizeof(unsigned char)));
  for (int i = 0; i < data_size; i++) {
    g_h[i] = std::rand() % 256;
    f_h[i] = std::rand() % 256;
  }

  iris::DMem M_d(M_h, 16 * sizeof(float));
  iris::DMem g_d(g_h, data_size * sizeof(unsigned char));
  iris::DMem f_d(f_h, data_size * sizeof(unsigned char));
  iris::DMem ivf_d(ivf_h, vol_size * sizeof(unsigned char));
  iris::DMem ivg_d(ivg_h, vol_size * sizeof(unsigned char));
  iris::DMem threshold_d(data_threshold_h, vol_size * sizeof(bool));

  size_t gws[1] = {static_cast<size_t>(NUM_BLOCKS * NUM_THREADS)};
  size_t lws[1] = {NUM_THREADS};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[9] = {&M_d, const_cast<int*>(&vol_size), &g_d, &f_d, &g_vol, &f_vol,
                       &ivf_d, &ivg_d, &threshold_d};
    int pinfo[9] = {iris_r, sizeof(vol_size), iris_r, iris_r, sizeof(g_vol), sizeof(f_vol),
                    iris_w, iris_w, iris_w};
    task.kernel("spm", 1, nullptr, gws, lws, 9, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(ivf_d);
      task.flush_out(ivg_d);
      task.flush_out(threshold_d);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  ivf_d.fetch(ivf_h);
  ivg_d.fetch(ivg_h);
  threshold_d.fetch(data_threshold_h);

  int count = 0;
  for (int i = 0; i < vol_size; i++) {
    if (data_threshold_h[i]) {
      hist_d[ivf_h[i] + ivg_h[i] * 256] += 1;
      count++;
    }
  }
  std::printf("Device count: %d\n", count);

  count = 0;
  spm_reference(M_h, vol_size, g_h, f_h, g_vol, f_vol, ivf_h, ivg_h, data_threshold_h);
  for (int i = 0; i < vol_size; i++) {
    if (data_threshold_h[i]) {
      hist_h[ivf_h[i] + ivg_h[i] * 256] += 1;
      count++;
    }
  }
  std::printf("Host count: %d\n", count);

  int max_diff = 0;
  for (int i = 0; i < 65536; i++) {
    if (hist_h[i] != hist_d[i])
      max_diff = std::max(max_diff, std::abs(hist_h[i] - hist_d[i]));
  }
  std::printf("Maximum difference %d\n", max_diff);

  std::free(hist_h);
  std::free(hist_d);
  std::free(ivf_h);
  std::free(ivg_h);
  std::free(g_h);
  std::free(f_h);
  std::free(data_threshold_h);

  platform.finalize();
  return 0;
}
