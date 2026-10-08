#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define RADIUS 7
#define BLOCK_SIZE 256

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <length> <repeat>\n", argv[0]);
    std::printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (length <= 0 || repeat <= 0 || length % BLOCK_SIZE != 0) {
    std::fprintf(stderr, "length must be a positive multiple of %d and repeat must be positive\n", BLOCK_SIZE);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const size_t size = static_cast<size_t>(length) * sizeof(int);
  const size_t pad_size = static_cast<size_t>(length + RADIUS) * sizeof(int);

  int* a = static_cast<int*>(std::malloc(pad_size));
  int* b = static_cast<int*>(std::malloc(size));
  if (!a || !b) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(a);
    std::free(b);
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < length + RADIUS; i++) a[i] = i;

  iris::DMem mem_a(a, pad_size);
  iris::DMem mem_b(b, size);

  const size_t gws[1] = {static_cast<size_t>(length)};
  const size_t lws[1] = {BLOCK_SIZE};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = {&mem_a, &mem_b};
    int pinfo[2] = {iris_r, iris_w};
    task.kernel("stencil_1d", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 2, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_b);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  mem_b.fetch(b);

  bool ok = true;
  for (int i = 0; i < 2 * RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i + 2 * RADIUS; j++) {
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    }
    if (s != b[i]) {
      std::printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2 * RADIUS; i < length && ok; i++) {
    int s = 0;
    for (int j = i - RADIUS; j <= i + RADIUS; j++) {
      s += a[j];
    }
    if (s != b[i]) {
      std::printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(a);
  std::free(b);
  platform.finalize();
  return ok ? 0 : 1;
}
