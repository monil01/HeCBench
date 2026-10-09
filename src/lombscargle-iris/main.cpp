#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void lombscargle_cpu(const int x_shape,
                            const int freqs_shape,
                            const float* __restrict x,
                            const float* __restrict y,
                            const float* __restrict freqs,
                            float* __restrict pgram,
                            const float y_dot)
{
  for (int tid = 0; tid < freqs_shape; tid++) {
    const float freq = freqs[tid];
    float xc = 0.0f;
    float xs = 0.0f;
    float cc = 0.0f;
    float ss = 0.0f;
    float cs = 0.0f;

    for (int j = 0; j < x_shape; j++) {
      float s;
      float c;
      sincosf(freq * x[j], &s, &c);
      xc += y[j] * c;
      xs += y[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;
    }

    float c_tau;
    float s_tau;
    const float tau = atan2f(2.0f * cs, cc - ss) / (2.0f * freq);
    sincosf(freq * tau, &s_tau, &c_tau);
    const float c_tau2 = c_tau * c_tau;
    const float s_tau2 = s_tau * s_tau;
    const float cs_tau = 2.0f * c_tau * s_tau;

    pgram[tid] = (0.5f * ((((c_tau * xc + s_tau * xs) * (c_tau * xc + s_tau * xs) /
                             (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) +
                            ((c_tau * xs - s_tau * xc) * (c_tau * xs - s_tau * xc) /
                             (c_tau2 * ss - cs_tau * cs + s_tau2 * cc))))) * y_dot;
  }
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
  const int x_shape = 1000;
  const int freqs_shape = 100000;
  const float A = 2.0f;
  const float w = 1.0f;
  const float phi = 1.57f;

  float* x = static_cast<float*>(std::malloc(sizeof(float) * x_shape));
  float* y = static_cast<float*>(std::malloc(sizeof(float) * x_shape));
  float* f = static_cast<float*>(std::malloc(sizeof(float) * freqs_shape));
  float* p = static_cast<float*>(std::malloc(sizeof(float) * freqs_shape));
  float* p2 = static_cast<float*>(std::malloc(sizeof(float) * freqs_shape));
  if (!x || !y || !f || !p || !p2) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < x_shape; i++) {
    x[i] = 0.01f + i * (31.4f - 0.01f) / x_shape;
    y[i] = A * sinf(w * x[i] + phi);
  }
  for (int i = 0; i < freqs_shape; i++) {
    f[i] = 0.01f + i * (10.0f - 0.01f) / freqs_shape;
    p[i] = 0.0f;
  }

  const float y_dot = 2.0f / 1.5f;
  iris::DMem mem_x(x, sizeof(float) * x_shape);
  iris::DMem mem_y(y, sizeof(float) * x_shape);
  iris::DMem mem_f(f, sizeof(float) * freqs_shape);
  iris::DMem mem_p(p, sizeof(float) * freqs_shape);

  const size_t lws[1] = {256};
  const size_t gws[1] = {static_cast<size_t>((freqs_shape + 255) / 256 * 256)};

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    int x_shape_arg = x_shape;
    int freqs_shape_arg = freqs_shape;
    iris::Task task;
    void* params[6] = {&x_shape_arg, &freqs_shape_arg, &mem_x, &mem_y, &mem_f, &mem_p};
    int pinfo[6] = {sizeof(x_shape_arg), sizeof(freqs_shape_arg), iris_r, iris_r, iris_r, iris_w};
    task.kernel("lombscargle_kernel", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 6, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  iris::Task flush;
  flush.flush_out(mem_p);
  flush.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();

  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (us)\n", (time * 1e-3) / repeat);

  mem_p.fetch(p);
  lombscargle_cpu(x_shape, freqs_shape, x, y, f, p2, y_dot);

  bool error = false;
  for (int i = 0; i < freqs_shape; i++) {
    if (std::fabs(p[i] - p2[i]) > 1e-1f) {
      std::printf("%.3f %.3f\n", p[i], p2[i]);
      error = true;
      break;
    }
  }

  std::printf("%s\n", error ? "FAIL" : "PASS");

  std::free(x);
  std::free(y);
  std::free(f);
  std::free(p);
  std::free(p2);
  platform.finalize();
  return error ? 1 : 0;
}
