#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define NUM_SIZE 16
#define BLOCK_SIZE 256

static void setup(size_t* size)
{
  for (int i = 0; i < NUM_SIZE; i++) {
    size[i] = static_cast<size_t>(1) << (i + 6);
  }
}

static void valSet(int* A, int val, size_t size)
{
  const size_t len = size / sizeof(int);
  for (size_t i = 0; i < len; i++) {
    A[i] = val;
  }
}

static bool checkValue(const int* A, int val, size_t size)
{
  const size_t len = size / sizeof(int);
  for (size_t i = 0; i < len; i++) {
    if (A[i] != val) return false;
  }
  return true;
}

static void submitTouch(iris::DMem& mem_A, int n, int mode, bool flush)
{
  const size_t gws[1] = {static_cast<size_t>(((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)};
  const size_t lws[1] = {BLOCK_SIZE};
  iris::Task task;
  void* params[2] = {&mem_A, &n};
  int pinfo[2] = {mode, sizeof(n)};
  task.kernel("memcpy_touch_kernel", 1, nullptr, const_cast<size_t*>(gws),
              const_cast<size_t*>(lws), 2, params, pinfo);
  if (flush) task.flush_out(mem_A);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <repeat>\n";
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  if (repeat <= 0) {
    std::cerr << "repeat must be positive\n";
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  size_t size[NUM_SIZE];
  setup(size);

  for (int i = 0; i < NUM_SIZE; i++) {
    int* A = static_cast<int*>(std::malloc(size[i]));
    if (A == nullptr) {
      std::cerr << "Host memory allocation failed\n";
      platform.finalize();
      return -1;
    }
    valSet(A, 1, size[i]);

    iris::DMem mem_A(A, size[i]);
    const int n = static_cast<int>(size[i] / sizeof(int));

    mem_A.update(A);
    for (int j = 0; j < repeat; j++) {
      submitTouch(mem_A, n, iris_r, false);
    }

    mem_A.update(A);
    auto start = std::chrono::steady_clock::now();
    for (int j = 0; j < repeat; j++) {
      submitTouch(mem_A, n, iris_r, false);
    }
    auto end = std::chrono::steady_clock::now();
    auto timeH2D = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Copy " << size[i] << " bytes from host to device takes "
              << (timeH2D * 1e-3f) / repeat <<  " us" << std::endl;

    valSet(A, 0, size[i]);

    for (int j = 0; j < repeat; j++) {
      submitTouch(mem_A, n, iris_rw, true);
    }

    valSet(A, 0, size[i]);
    start = std::chrono::steady_clock::now();
    for (int j = 0; j < repeat; j++) {
      submitTouch(mem_A, n, iris_rw, true);
    }
    end = std::chrono::steady_clock::now();
    auto timeD2H = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Copy " << size[i] << " bytes from device to host takes "
              << (timeD2H * 1e-3f) / repeat <<  " us" << std::endl;

    const bool ok = checkValue(A, 1, size[i]);
    std::free(A);

    std::cout << "Timing gap in nanoseconds per byte: "
              << static_cast<float>(std::abs(timeH2D - timeD2H)) / (repeat * size[i]);
    std::cout << std::endl << std::endl;

    if (!ok) {
      std::cerr << "FAIL\n";
      platform.finalize();
      return 1;
    }
  }

  std::cout << "PASS\n";
  platform.finalize();
  return 0;
}
