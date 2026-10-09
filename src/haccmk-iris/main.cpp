#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void haccmk_gold(int count1, float xxi, float yyi, float zzi,
                        float fsrrmax2, float mp_rsm2,
                        float* __restrict__ xx1,
                        float* __restrict__ yy1,
                        float* __restrict__ zz1,
                        float* __restrict__ mass1,
                        float* __restrict__ dxi,
                        float* __restrict__ dyi,
                        float* __restrict__ dzi)
{
  const float ma0 = 0.269327f;
  const float ma1 = -0.0750978f;
  const float ma2 = 0.0114808f;
  const float ma3 = -0.00109313f;
  const float ma4 = 0.0000605491f;
  const float ma5 = -0.00000147177f;
  float xi = 0.0f;
  float yi = 0.0f;
  float zi = 0.0f;

  for (int j = 0; j < count1; j++) {
    const float dxc = xx1[j] - xxi;
    const float dyc = yy1[j] - yyi;
    const float dzc = zz1[j] - zzi;
    const float r2 = dxc * dxc + dyc * dyc + dzc * dzc;
    const float m = (r2 < fsrrmax2) ? mass1[j] : 0.0f;
    const float ftmp = r2 + mp_rsm2;
    const float f = m * (1.0f / (ftmp * std::sqrt(ftmp)) -
                         (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))));
    xi += f * dxc;
    yi += f * dyc;
    zi += f * dzc;
  }

  *dxi = xi;
  *dyi = yi;
  *dzi = zi;
}

static void haccmk_iris(int repeat, int n1, int n2, float* xx, float* yy,
                        float* zz, float* mass, float* vx2, float* vy2,
                        float* vz2, float fsrmax, float mp_rsm, float fcoeff)
{
  const int block_size = 256;
  float total_time = 0.0f;

  for (int iter = 0; iter < repeat; iter++) {
    for (int i = 0; i < n1; i++) {
      vx2[i] = 0.0f;
      vy2[i] = 0.0f;
      vz2[i] = 0.0f;
    }

    iris::DMem mem_xx(xx, sizeof(float) * n2);
    iris::DMem mem_yy(yy, sizeof(float) * n2);
    iris::DMem mem_zz(zz, sizeof(float) * n2);
    iris::DMem mem_mass(mass, sizeof(float) * n2);
    iris::DMem mem_vx2(vx2, sizeof(float) * n1);
    iris::DMem mem_vy2(vy2, sizeof(float) * n1);
    iris::DMem mem_vz2(vz2, sizeof(float) * n1);

    const size_t gws[1] = { static_cast<size_t>(((n1 + block_size - 1) / block_size) * block_size) };
    const size_t lws[1] = { static_cast<size_t>(block_size) };
    void* params[13] = { &n1, &n2, &mem_xx, &mem_yy, &mem_zz, &mem_mass,
                         &mem_vx2, &mem_vy2, &mem_vz2, &fsrmax, &mp_rsm, &fcoeff,
                         const_cast<int*>(&block_size) };
    int pinfo[13] = { sizeof(n1), sizeof(n2), iris_r, iris_r, iris_r, iris_r,
                      iris_rw, iris_rw, iris_rw, sizeof(fsrmax), sizeof(mp_rsm),
                      sizeof(fcoeff), sizeof(block_size) };

    const auto start = std::chrono::steady_clock::now();
    iris::Task task;
    task.kernel("haccmk_kernel", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 13, params, pinfo);
    task.flush_out(mem_vx2);
    task.flush_out(mem_vy2);
    task.flush_out(mem_vz2);
    task.submit(iris_gpu, nullptr, 1);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += static_cast<float>(elapsed);
  }

  std::printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int n1 = 784;
  const int n2 = 15000;
  std::printf("Outer loop count is set %d\n", n1);
  std::printf("Inner loop count is set %d\n", n2);

  float* xx = static_cast<float*>(std::malloc(sizeof(float) * n2));
  float* yy = static_cast<float*>(std::malloc(sizeof(float) * n2));
  float* zz = static_cast<float*>(std::malloc(sizeof(float) * n2));
  float* mass = static_cast<float*>(std::malloc(sizeof(float) * n2));
  float* vx2 = static_cast<float*>(std::malloc(sizeof(float) * n2));
  float* vy2 = static_cast<float*>(std::malloc(sizeof(float) * n2));
  float* vz2 = static_cast<float*>(std::malloc(sizeof(float) * n2));
  float* vx2_hw = static_cast<float*>(std::malloc(sizeof(float) * n2));
  float* vy2_hw = static_cast<float*>(std::malloc(sizeof(float) * n2));
  float* vz2_hw = static_cast<float*>(std::malloc(sizeof(float) * n2));

  if (!xx || !yy || !zz || !mass || !vx2 || !vy2 || !vz2 || !vx2_hw || !vy2_hw || !vz2_hw) {
    std::fprintf(stderr, "allocation failure\n");
    platform.finalize();
    return 1;
  }

  const float fcoeff = 0.23f;
  const float fsrrmax2 = 0.5f;
  const float mp_rsm2 = 0.03f;
  const float dx1 = 1.0f / static_cast<float>(n2);
  const float dy1 = 2.0f / static_cast<float>(n2);
  const float dz1 = 3.0f / static_cast<float>(n2);
  xx[0] = 0.0f;
  yy[0] = 0.0f;
  zz[0] = 0.0f;
  mass[0] = 2.0f;

  for (int i = 1; i < n2; i++) {
    xx[i] = xx[i - 1] + dx1;
    yy[i] = yy[i - 1] + dy1;
    zz[i] = zz[i - 1] + dz1;
    mass[i] = static_cast<float>(i) * 0.01f + xx[i];
  }

  for (int i = 0; i < n2; i++) {
    vx2[i] = 0.0f;
    vy2[i] = 0.0f;
    vz2[i] = 0.0f;
    vx2_hw[i] = 0.0f;
    vy2_hw[i] = 0.0f;
    vz2_hw[i] = 0.0f;
  }

  for (int i = 0; i < n1; ++i) {
    float dx2, dy2, dz2;
    haccmk_gold(n2, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, &dx2, &dy2, &dz2);
    vx2[i] += dx2 * fcoeff;
    vy2[i] += dy2 * fcoeff;
    vz2[i] += dz2 * fcoeff;
  }

  haccmk_iris(repeat, n1, n2, xx, yy, zz, mass, vx2_hw, vy2_hw, vz2_hw,
              fsrrmax2, mp_rsm2, fcoeff);

  int error = 0;
  const float eps = 1.0f;
  for (int i = 0; i < n2; i++) {
    if (std::fabs(vx2[i] - vx2_hw[i]) > eps) {
      std::printf("error at vx2[%d] %f %f\n", i, vx2[i], vx2_hw[i]);
      error = 1;
      break;
    }
    if (std::fabs(vy2[i] - vy2_hw[i]) > eps) {
      std::printf("error at vy2[%d]: %f %f\n", i, vy2[i], vy2_hw[i]);
      error = 1;
      break;
    }
    if (std::fabs(vz2[i] - vz2_hw[i]) > eps) {
      std::printf("error at vz2[%d]: %f %f\n", i, vz2[i], vz2_hw[i]);
      error = 1;
      break;
    }
  }

  std::free(xx);
  std::free(yy);
  std::free(zz);
  std::free(mass);
  std::free(vx2);
  std::free(vy2);
  std::free(vz2);
  std::free(vx2_hw);
  std::free(vy2_hw);
  std::free(vz2_hw);

  std::printf("%s\n", error ? "FAIL" : "PASS");
  platform.finalize();
  return error ? 1 : 0;
}
