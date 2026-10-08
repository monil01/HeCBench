#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "kernel.h"

#define REPEAT 1000
#define N 370
#define LDAT N
#define INCX 1
#define INCY 1
#define AT_SIZE (N * LDAT)
#define X_SIZE (N * INCX)
#define Y_SIZE (N * INCY)

static void chemv_cpu(float alpha_re, float alpha_im, float beta_re, float beta_im,
                      ComplexFloat AT[AT_SIZE], ComplexFloat X[X_SIZE],
                      ComplexFloat Y[Y_SIZE])
{
  for (int i0 = 0; i0 <= (N - 1); i0 += 1) {
    float var5_Re = ((Y[i0 * INCY].Re * beta_re) - (Y[i0 * INCY].Im * beta_im));
    float var5_Im = ((Y[i0 * INCY].Im * beta_re) + (Y[i0 * INCY].Re * beta_im));
    Y[i0 * INCY].Re = var5_Re;
    Y[i0 * INCY].Im = var5_Im;
  }

  for (int i1 = 0; i1 <= ((N - 1) + 1) - 1; i1 += 1) {
    float var2_Re = (alpha_re * AT[i1 * LDAT + i1].Re);
    float var2_Im = (alpha_im * AT[i1 * LDAT + i1].Re);
    float var3_Re = ((var2_Re * X[i1 * INCX].Re) - (var2_Im * X[i1 * INCX].Im));
    float var3_Im = ((var2_Im * X[i1 * INCX].Re) + (var2_Re * X[i1 * INCX].Im));
    Y[i1 * INCY].Re += var3_Re;
    Y[i1 * INCY].Im += var3_Im;
  }

  for (int i2 = 0; i2 <= ((N - 1) - 1); i2 += 1) {
    for (int i3 = 0; i3 <= (N - 1) - (1 + i2); i3 += 1) {
      float var94_Re = ((alpha_re * AT[i2 * LDAT + ((1 + i2) + i3)].Re) -
          (alpha_im * (-AT[i2 * LDAT + ((1 + i2) + i3)].Im)));
      float var94_Im = ((alpha_im * AT[i2 * LDAT + ((1 + i2) + i3)].Re) +
          (alpha_re * (-AT[i2 * LDAT + ((1 + i2) + i3)].Im)));
      float var95_Re = ((var94_Re * X[((i3 + i2) + 1) * INCX].Re) -
          (var94_Im * X[((i3 + i2) + 1) * INCX].Im));
      float var95_Im = ((var94_Im * X[((i3 + i2) + 1) * INCX].Re) +
          (var94_Re * X[((i3 + i2) + 1) * INCX].Im));
      Y[i2 * INCY].Re += var95_Re;
      Y[i2 * INCY].Im += var95_Im;

      float var97_Re = ((alpha_re * AT[i2 * LDAT + ((1 + i2) + i3)].Re) -
          (alpha_im * AT[i2 * LDAT + ((1 + i2) + i3)].Im));
      float var97_Im = ((alpha_im * AT[i2 * LDAT + ((1 + i2) + i3)].Re) +
          (alpha_re * AT[i2 * LDAT + ((1 + i2) + i3)].Im));
      float var98_Re = ((var97_Re * X[i2 * INCX].Re) -
          (var97_Im * X[i2 * INCX].Im));
      float var98_Im = ((var97_Im * X[i2 * INCX].Re) +
          (var97_Re * X[i2 * INCX].Im));
      Y[((i3 + i2) + 1) * INCY].Re += var98_Re;
      Y[((i3 + i2) + 1) * INCY].Im += var98_Im;
    }
  }
}

static void chemv_iris(float alpha_re, float alpha_im, float beta_re, float beta_im,
                       ComplexFloat AT[AT_SIZE], ComplexFloat X[X_SIZE],
                       ComplexFloat Y[Y_SIZE])
{
  iris::DMem dev_AT(AT, AT_SIZE * sizeof(ComplexFloat));
  iris::DMem dev_X(X, X_SIZE * sizeof(ComplexFloat));
  iris::DMem dev_Y(Y, Y_SIZE * sizeof(ComplexFloat));

  size_t gws[1] = {12 * 32};
  size_t lws[1] = {32};

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < REPEAT; n++) {
    {
      iris::Task task;
      void* params[7] = {&dev_AT, &dev_X, &dev_Y, &alpha_im, &alpha_re, &beta_im, &beta_re};
      int pinfo[7] = {iris_r, iris_r, iris_rw, sizeof(alpha_im), sizeof(alpha_re),
                      sizeof(beta_im), sizeof(beta_re)};
      task.kernel("chemv_kernel0", 1, nullptr, gws, lws, 7, params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
    }
    {
      iris::Task task;
      void* params[5] = {&dev_AT, &dev_X, &dev_Y, &alpha_im, &alpha_re};
      int pinfo[5] = {iris_r, iris_r, iris_rw, sizeof(alpha_im), sizeof(alpha_re)};
      task.kernel("chemv_kernel1", 1, nullptr, gws, lws, 5, params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
    }
  }
  iris::Task flush;
  flush.flush_out(dev_Y);
  flush.submit(iris_gpu, nullptr, 1);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of chemv kernels: %f (us)\n", (time * 1e-3f) / REPEAT);

  dev_Y.fetch(Y);
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  ComplexFloat AT[AT_SIZE];
  ComplexFloat X[X_SIZE];
  ComplexFloat Y_cpu[Y_SIZE];
  ComplexFloat Y_gpu[Y_SIZE];

  for (int i = 0; i < N; i++) {
    X[i * INCX] = ComplexFloat{static_cast<float>(i + 5), static_cast<float>(i * 2)};
    Y_cpu[i * INCY] = ComplexFloat{static_cast<float>(i * 3), static_cast<float>(i + 7)};
    Y_gpu[i * INCY] = ComplexFloat{static_cast<float>(i * 3), static_cast<float>(i + 7)};
    for (int j = 0; j < LDAT; j++) {
      AT[i * LDAT + j] = ComplexFloat{static_cast<float>(i + j), static_cast<float>(i + 3)};
    }
  }

  const float alpha_re = 3.14f;
  const float alpha_im = 1.59f;
  const float beta_re = 2.71f;
  const float beta_im = 8.28f;

  chemv_cpu(alpha_re, alpha_im, beta_re, beta_im, AT, X, Y_cpu);
  chemv_iris(alpha_re, alpha_im, beta_re, beta_im, AT, X, Y_gpu);

  for (int i = 0; i < N; i++) {
    if ((std::fabs(Y_cpu[i * INCY].Re - Y_gpu[i * INCY].Re) > 1e-3f) ||
        (std::fabs(Y_cpu[i * INCY].Im - Y_gpu[i * INCY].Im) > 1e-3f)) {
      std::printf("%d %f %f\n", i, Y_cpu[i * INCY].Re, Y_gpu[i * INCY].Re);
      std::printf("FAIL\n");
      platform.finalize();
      return EXIT_FAILURE;
    }
  }

  std::printf("PASS\n");
  platform.finalize();
  return EXIT_SUCCESS;
}
