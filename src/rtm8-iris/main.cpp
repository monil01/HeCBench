#include <iris/iris.hpp>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define nx 680
#define ny 134
#define nz 450

inline int indexTo1D(int x, int y, int z) { return x + y * nx + z * nx * ny; }

void rtm8_cpu(float* vsq, float* current_s, float* current_r, float* next_s,
              float* next_r, float* image, float* a, size_t) {
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (int z = 4; z < nz - 4; z++) {
    for (int y = 4; y < ny - 4; y++) {
      for (int x = 4; x < nx - 4; x++) {
        float div =
            a[0] * current_s[indexTo1D(x, y, z)] +
            a[1] * (current_s[indexTo1D(x + 1, y, z)] +
                    current_s[indexTo1D(x - 1, y, z)] +
                    current_s[indexTo1D(x, y + 1, z)] +
                    current_s[indexTo1D(x, y - 1, z)] +
                    current_s[indexTo1D(x, y, z + 1)] +
                    current_s[indexTo1D(x, y, z - 1)]) +
            a[2] * (current_s[indexTo1D(x + 2, y, z)] +
                    current_s[indexTo1D(x - 2, y, z)] +
                    current_s[indexTo1D(x, y + 2, z)] +
                    current_s[indexTo1D(x, y - 2, z)] +
                    current_s[indexTo1D(x, y, z + 2)] +
                    current_s[indexTo1D(x, y, z - 2)]) +
            a[3] * (current_s[indexTo1D(x + 3, y, z)] +
                    current_s[indexTo1D(x - 3, y, z)] +
                    current_s[indexTo1D(x, y + 3, z)] +
                    current_s[indexTo1D(x, y - 3, z)] +
                    current_s[indexTo1D(x, y, z + 3)] +
                    current_s[indexTo1D(x, y, z - 3)]) +
            a[4] * (current_s[indexTo1D(x + 4, y, z)] +
                    current_s[indexTo1D(x - 4, y, z)] +
                    current_s[indexTo1D(x, y + 4, z)] +
                    current_s[indexTo1D(x, y - 4, z)] +
                    current_s[indexTo1D(x, y, z + 4)] +
                    current_s[indexTo1D(x, y, z - 4)]);

        next_s[indexTo1D(x, y, z)] =
            2 * current_s[indexTo1D(x, y, z)] - next_s[indexTo1D(x, y, z)] +
            vsq[indexTo1D(x, y, z)] * div;
        div =
            a[0] * current_r[indexTo1D(x, y, z)] +
            a[1] * (current_r[indexTo1D(x + 1, y, z)] +
                    current_r[indexTo1D(x - 1, y, z)] +
                    current_r[indexTo1D(x, y + 1, z)] +
                    current_r[indexTo1D(x, y - 1, z)] +
                    current_r[indexTo1D(x, y, z + 1)] +
                    current_r[indexTo1D(x, y, z - 1)]) +
            a[2] * (current_r[indexTo1D(x + 2, y, z)] +
                    current_r[indexTo1D(x - 2, y, z)] +
                    current_r[indexTo1D(x, y + 2, z)] +
                    current_r[indexTo1D(x, y - 2, z)] +
                    current_r[indexTo1D(x, y, z + 2)] +
                    current_r[indexTo1D(x, y, z - 2)]) +
            a[3] * (current_r[indexTo1D(x + 3, y, z)] +
                    current_r[indexTo1D(x - 3, y, z)] +
                    current_r[indexTo1D(x, y + 3, z)] +
                    current_r[indexTo1D(x, y - 3, z)] +
                    current_r[indexTo1D(x, y, z + 3)] +
                    current_r[indexTo1D(x, y, z - 3)]) +
            a[4] * (current_r[indexTo1D(x + 4, y, z)] +
                    current_r[indexTo1D(x - 4, y, z)] +
                    current_r[indexTo1D(x, y + 4, z)] +
                    current_r[indexTo1D(x, y - 4, z)] +
                    current_r[indexTo1D(x, y, z + 4)] +
                    current_r[indexTo1D(x, y, z - 4)]);

        next_r[indexTo1D(x, y, z)] =
            2 * current_r[indexTo1D(x, y, z)] - next_r[indexTo1D(x, y, z)] +
            vsq[indexTo1D(x, y, z)] * div;
        image[indexTo1D(x, y, z)] =
            next_s[indexTo1D(x, y, z)] * next_r[indexTo1D(x, y, z)];
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int ArraySize = nx * ny * nz;
  std::vector<float> next_s(ArraySize, 0.0f);
  std::vector<float> current_s(ArraySize, 1.0f);
  std::vector<float> next_r(ArraySize, 0.0f);
  std::vector<float> current_r(ArraySize, 1.0f);
  std::vector<float> vsq(ArraySize, 1.0f);
  std::vector<float> image_gpu(ArraySize, 0.5f);
  std::vector<float> image_cpu(ArraySize, 0.5f);
  float a[5] = {-1.f / 560.f, 8.f / 315.f, -0.2f, 1.6f, -1435.f / 504.f};

  double memory = ArraySize * sizeof(float) * 6.0;
  double pts = static_cast<double>(repeat) * (nx - 8) * (ny - 8) * (nz - 8);
  double flops = 67 * pts;
  std::printf("memory (MB) = %lf\n", memory / 1e6);
  std::printf("pts (billions) = %lf\n", pts / 1e9);
  std::printf("Tflops = %lf\n", flops / 1e12);

  iris::DMem vsq_d(vsq.data(), ArraySize * sizeof(float));
  iris::DMem next_s_d(next_s.data(), ArraySize * sizeof(float));
  iris::DMem current_s_d(current_s.data(), ArraySize * sizeof(float));
  iris::DMem next_r_d(next_r.data(), ArraySize * sizeof(float));
  iris::DMem current_r_d(current_r.data(), ArraySize * sizeof(float));
  iris::DMem image_d(image_gpu.data(), ArraySize * sizeof(float));
  iris::DMem a_d(a, 5 * sizeof(float));

  int groupSize = 16;
  int nx_pad = (nx + groupSize - 1) / groupSize;
  int ny_pad = (ny + groupSize - 1) / groupSize;
  int nz_pad = nz;
  size_t gws[] = {static_cast<size_t>(nx_pad * groupSize),
                  static_cast<size_t>(ny_pad * groupSize),
                  static_cast<size_t>(nz_pad)};
  size_t lws[] = {static_cast<size_t>(groupSize), static_cast<size_t>(groupSize),
                  1};
  size_t N = ArraySize;

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < repeat; t++) {
    iris::Task task;
    void* params[] = {&vsq_d, &current_s_d, &next_s_d, &current_r_d,
                      &next_r_d, &image_d, &a_d, &N};
    int pinfo[] = {iris_r, iris_r, iris_rw, iris_rw,
                   iris_rw, iris_rw, iris_r, sizeof(N)};
    task.kernel("rtm8", 3, nullptr, gws, lws, 8, params, pinfo);
    task.flush_out(image_d);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double dt = time * 1e-9;

  start = std::chrono::steady_clock::now();
  for (int t = 0; t < repeat; t++) {
    rtm8_cpu(vsq.data(), current_s.data(), next_s.data(), current_r.data(),
             next_r.data(), image_cpu.data(), a, ArraySize);
  }
  end = std::chrono::steady_clock::now();
  time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double ht = time * 1e-9;

  bool ok = true;
  for (int i = 0; i < ArraySize; i++) {
    if (std::fabs(image_cpu[i] - image_gpu[i]) > 0.1f) {
      std::printf("@index %d host: %f device %f\n", i, image_cpu[i],
                  image_gpu[i]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  double pt_rate = pts / dt;
  double flop_rate = flops / dt;
  double speedup = ht / dt;
  std::printf("dt = %lf (sec)\n", dt);
  std::printf("pt_rate (millions/sec) = %lf\n", pt_rate / 1e6);
  std::printf("flop_rate (Gflops) = %lf\n", flop_rate / 1e9);
  std::printf("speedup over cpu = %lf\n", speedup);
  std::printf("average kernel execution time = %lf (s)\n", dt / repeat);

  platform.finalize();
  return 0;
}
