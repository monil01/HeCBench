#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define RY 15
#define YG 6
#define GC 4
#define CB 11
#define BM 13
#define MR 6
#define MAXCOLS (RY + YG + GC + CB + BM + MR)
using uchar = unsigned char;

static void setcols(int cw[MAXCOLS][3], int r, int g, int b, int k) {
  cw[k][0] = r;
  cw[k][1] = g;
  cw[k][2] = b;
}

static void computeColor(float fx, float fy, uchar* pix) {
  int cw[MAXCOLS][3];
  int k = 0;
  for (int i = 0; i < RY; i++) setcols(cw, 255, 255 * i / RY, 0, k++);
  for (int i = 0; i < YG; i++) setcols(cw, 255 - 255 * i / YG, 255, 0, k++);
  for (int i = 0; i < GC; i++) setcols(cw, 0, 255, 255 * i / GC, k++);
  for (int i = 0; i < CB; i++) setcols(cw, 0, 255 - 255 * i / CB, 255, k++);
  for (int i = 0; i < BM; i++) setcols(cw, 255 * i / BM, 0, 255, k++);
  for (int i = 0; i < MR; i++) setcols(cw, 255, 0, 255 - 255 * i / MR, k++);

  float rad = sqrtf(fx * fx + fy * fy);
  float a = atan2f(-fy, -fx) / static_cast<float>(M_PI);
  float fk = (a + 1.f) / 2.f * (MAXCOLS - 1);
  int k0 = static_cast<int>(fk);
  int k1 = (k0 + 1) % MAXCOLS;
  float f = fk - k0;
  for (int b = 0; b < 3; b++) {
    float col0 = cw[k0][b] / 255.f;
    float col1 = cw[k1][b] / 255.f;
    float col = (1.f - f) * col0 + f * col1;
    if (rad <= 1)
      col = 1.f - rad * (1.f - col);
    else
      col *= .75f;
    pix[2 - b] = static_cast<int>(255.f * col);
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::printf("Usage: %s <range> <size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const float truerange = std::atof(argv[1]);
  const int size = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  const float range = 1.04f * truerange;
  const int half_size = size / 2;
  const size_t imgSize = static_cast<size_t>(size) * size * 3;

  uchar* pix = static_cast<uchar*>(std::malloc(imgSize));
  uchar* res = static_cast<uchar*>(std::malloc(imgSize));
  if (!pix || !res) {
    std::fprintf(stderr, "host allocation failed\n");
    return 1;
  }
  std::memset(pix, 0, imgSize);
  std::memset(res, 0, imgSize);

  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      float fx = static_cast<float>(x) / static_cast<float>(half_size) * range - range;
      float fy = static_cast<float>(y) / static_cast<float>(half_size) * range - range;
      if (x == half_size || y == half_size) continue;
      size_t idx = (static_cast<size_t>(y) * size + x) * 3;
      computeColor(fx / truerange, fy / truerange, pix + idx);
    }
  }

  std::printf("Start execution on a device\n");
  iris::DMem mem_res(res, imgSize);

  size_t gws[] = {static_cast<size_t>(((size + 15) / 16) * 16),
                  static_cast<size_t>(((size + 15) / 16) * 16)};
  size_t lws[] = {16, 16};
  void* params[] = {&mem_res, (void*)&size, (void*)&half_size, (void*)&range,
                    (void*)&truerange};
  int pinfo[] = {iris_w, sizeof(size), sizeof(half_size), sizeof(range),
                 sizeof(truerange)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    task.kernel("color", 2, nullptr, gws, lws, 5, params, pinfo);
    task.flush_out(mem_res);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time : %f (ms)\n",
              (time * 1e-6f) / repeat);

  const int tolerance = 1;
  int fail = 0;
  int max_error = 0;
  for (size_t i = 0; i < imgSize; i++) {
    int e = std::abs(static_cast<int>(res[i]) - static_cast<int>(pix[i]));
    if (e > tolerance) {
      fail = 1;
      if (e > max_error) max_error = e;
    }
  }
  if (fail)
    std::printf("Verification failed. Maximum error between host and device results: %d\n",
                max_error);
  else
    std::printf("PASS\n");

  std::free(pix);
  std::free(res);
  platform.finalize();
  return fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
