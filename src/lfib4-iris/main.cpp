#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define P1 55
#define P2 119
#define P3 179
#define P4 256

#define LKNB 8

typedef unsigned int uint32_t;

static void LFIB4(uint32_t n, uint32_t* x)
{
  for (uint32_t k = P4; k < n; k++) {
    x[k] = x[k - P1] + x[k - P2] + x[k - P3] + x[k - P4];
  }
}

static void gLFIB4(uint32_t* x, int s, int r, const uint32_t* seed)
{
  for (int i = 0; i < P4; i++) {
    x[i] = seed[i];
  }

  std::vector<uint32_t> y(3 * P4, 0);
  y[2 * P4] = 1;

  iris::DMem mem_x(x, static_cast<size_t>(r) * s * sizeof(uint32_t));
  iris::DMem mem_y(y.data(), y.size() * sizeof(uint32_t));

  {
    size_t gws[1] = { P4 };
    size_t lws[1] = { P4 };
    iris::Task task;
    void* params[2] = { &mem_x, &s };
    int pinfo[2] = { iris_rw, sizeof(s) };
    task.kernel("firstColGPU", 1, nullptr, gws, lws, 2, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  {
    size_t gws[1] = { P4 };
    size_t lws[1] = { P4 };
    iris::Task task;
    void* params[2] = { &mem_y, &s };
    int pinfo[2] = { iris_rw, sizeof(s) };
    task.kernel("colYGPU", 1, nullptr, gws, lws, 2, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  {
    size_t gws[1] = { 2 * P4 };
    size_t lws[1] = { 2 * P4 };
    iris::Task task;
    void* params[4] = { &mem_x, &mem_y, &s, &r };
    int pinfo[4] = { iris_rw, iris_r, sizeof(s), sizeof(r) };
    task.kernel("lastEntGPU", 1, nullptr, gws, lws, 4, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  {
    const int blocks = r / LKNB + (r % LKNB ? 1 : 0);
    size_t gws[1] = { static_cast<size_t>(blocks) * P4 };
    size_t lws[1] = { P4 };
    iris::Task task;
    void* params[3] = { &mem_x, &s, &r };
    int pinfo[3] = { iris_rw, sizeof(s), sizeof(r) };
    task.kernel("colsGPU", 1, nullptr, gws, lws, 3, params, pinfo);
    task.flush_out(mem_x);
    task.submit(iris_gpu, nullptr, 1);
  }
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::printf("Usage: ./main <n>\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const uint32_t n = static_cast<uint32_t>(std::atoi(argv[1]));
  std::vector<uint32_t> x(n);

  int failures = 0;
  for (uint32_t r = 16; r <= 4096; r = r * 2) {
    uint32_t s = n / r;
    s -= (s % 256 == 0 ? 0 : s % 256);
    while (s * r < n) {
      r++;
    }

    std::printf("n=%d r=%d s=%d\n", n, r, s);
    std::vector<uint32_t> z(static_cast<size_t>(r) * s);

    std::srand(1234);
    for (uint32_t k = 0; k < P4; k++) {
      x[k] = z[k] = static_cast<uint32_t>(std::rand());
    }

    auto start = std::chrono::steady_clock::now();
    LFIB4(n, x.data());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> host_time = end - start;

    start = std::chrono::steady_clock::now();
    gLFIB4(z.data(), static_cast<int>(s), static_cast<int>(r), z.data());
    end = std::chrono::steady_clock::now();
    std::chrono::duration<float> device_time = end - start;

    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
      if (x[i] != z[i]) {
        ok = false;
        break;
      }
    }

    std::printf("r = %d | host time = %lf | device time = %lf | speedup = %.1f ",
                r, host_time.count(), device_time.count(), host_time.count() / device_time.count());
    std::printf("check = %s\n", ok ? "PASS" : "FAIL");
    failures += ok ? 0 : 1;
  }

  platform.finalize();
  return failures == 0 ? 0 : 1;
}
