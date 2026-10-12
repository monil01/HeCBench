#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

typedef unsigned long ulong;

static ulong* convertBuffer2Array(char* cbuffer, unsigned size, unsigned step)
{
  unsigned i, j;
  ulong* values = nullptr;
  posix_memalign((void**)&values, 1024, sizeof(ulong) * size / step);
  for (i = 0; i < size / step; i++) values[i] = 0;
  for (i = 0; i < size; i += step) {
    for (j = 0; j < step; j++) {
      values[i / step] += (ulong)((unsigned char)cbuffer[i + j]) << (8 * j);
    }
  }
  return values;
}

static unsigned my_abs(int x)
{
  unsigned t = x >> 31;
  return (x ^ t) - t;
}

static unsigned FPCCompress(ulong* values, unsigned size)
{
  unsigned compressable = 0;
  for (unsigned i = 0; i < size; i++) {
    if (values[i] == 0) {
      compressable += 1;
    } else if (my_abs((int)(values[i])) <= 0xFF) {
      compressable += 1;
    } else if (my_abs((int)(values[i])) <= 0xFFFF) {
      compressable += 2;
    } else if (((values[i]) & 0xFFFF) == 0) {
      compressable += 2;
    } else if (my_abs((int)((values[i]) & 0xFFFF)) <= 0xFF &&
               my_abs((int)((values[i] >> 16) & 0xFFFF)) <= 0xFF) {
      compressable += 2;
    } else {
      unsigned byte0 = (values[i]) & 0xFF;
      unsigned byte1 = (values[i] >> 8) & 0xFF;
      unsigned byte2 = (values[i] >> 16) & 0xFF;
      unsigned byte3 = (values[i] >> 24) & 0xFF;
      compressable += (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) ? 1 : 4;
    }
  }
  return compressable;
}

static void run_fpc_kernel(const char* kernel_name, const ulong* values,
                           unsigned* cmp_size_hw, int values_size, int wgs)
{
  *cmp_size_hw = 0;
  iris::DMem d_values(const_cast<ulong*>(values), values_size * sizeof(ulong));
  iris::DMem d_cmp_size(cmp_size_hw, sizeof(unsigned));

  size_t gws[1] = {static_cast<size_t>((values_size / wgs) * wgs)};
  size_t lws[1] = {static_cast<size_t>(wgs)};
  iris::Task task;
  void* params[2] = {&d_values, &d_cmp_size};
  int pinfo[2] = {iris_r, iris_rw};
  task.kernel(kernel_name, 1, nullptr, gws, lws, 2, params, pinfo);
  task.flush_out(d_cmp_size);
  task.submit(iris_gpu, nullptr, 1);
  d_cmp_size.fetch(cmp_size_hw);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <work-group size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int wgs = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int step = 4;
  const size_t size = (size_t)wgs * wgs * wgs;
  char* cbuffer = (char*)malloc(size * step);

  srand(2);
  for (size_t i = 0; i < size * step; i++) cbuffer[i] = 0xFF << (rand() % 256);

  ulong* values = convertBuffer2Array(cbuffer, size, step);
  unsigned values_size = size / step;
  unsigned cmp_size = FPCCompress(values, values_size);
  unsigned cmp_size_hw;

  bool ok = true;

  run_fpc_kernel("fpc_kernel", values, &cmp_size_hw, values_size, wgs);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_fpc_kernel("fpc_kernel", values, &cmp_size_hw, values_size, wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  run_fpc_kernel("fpc2_kernel", values, &cmp_size_hw, values_size, wgs);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_fpc_kernel("fpc2_kernel", values, &cmp_size_hw, values_size, wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc2 failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }
  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc2: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  printf("%s\n", ok ? "PASS" : "FAIL");

  free(values);
  free(cbuffer);
  platform.finalize();
  return 0;
}
