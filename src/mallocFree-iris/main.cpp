#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#define NUM_SIZE 19
#define NUM_ITER 500

#define Clock() std::chrono::steady_clock::now()

static void valSet(int* A, int val, size_t size)
{
  const size_t len = size / sizeof(int);
  for (size_t i = 0; i < len; i++) {
    A[i] = val;
  }
}

static void setup(size_t* size, int& num, int** pA, const size_t totalGlobalMem)
{
  for (int i = 0; i < num; i++) {
    size[i] = static_cast<size_t>(1) << (i + 6);
    if ((NUM_ITER + 1) * size[i] > totalGlobalMem) {
      num = i;
      break;
    }
  }
  *pA = static_cast<int*>(std::malloc(size[num - 1]));
  valSet(*pA, 1, size[num - 1]);
}

static void testInit(size_t size)
{
  std::printf("Initial allocation and deallocation\n");

  std::vector<int> host(size / sizeof(int), 0);
  auto start = Clock();
  auto mem = std::make_unique<iris::DMem>(host.data(), size);
  auto end = Clock();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("iris::DMem(%zu) takes %lf us\n", size, time * 1e-3);

  start = Clock();
  mem.reset();
  end = Clock();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("iris::DMem free(%zu) takes %lf us\n", size, time * 1e-3);
  std::printf("\n");
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <total global memory size in bytes>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const size_t totalGlobalMem = std::atol(argv[1]);
  size_t size[NUM_SIZE] = { 0 };
  int num = NUM_SIZE;
  int* A = nullptr;
  setup(size, num, &A, totalGlobalMem);

  std::printf("\n==== Evaluate iris::DMem create and destroy ====\n");
  testInit(size[0]);

  std::vector<std::unique_ptr<std::vector<int>>> hosts(NUM_ITER);
  std::vector<std::unique_ptr<iris::DMem>> mems(NUM_ITER);

  for (int i = 0; i < num; i++) {
    const size_t ints = size[i] / sizeof(int);
    auto start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      hosts[j] = std::make_unique<std::vector<int>>(ints, 0);
      mems[j] = std::make_unique<iris::DMem>(hosts[j]->data(), size[i]);
    }
    auto end = Clock();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("iris::DMem(%zu) takes %lf us\n", size[i], time * 1e-3 / NUM_ITER);

    start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      mems[j].reset();
      hosts[j].reset();
    }
    end = Clock();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("iris::DMem free(%zu) takes %lf us\n", size[i], time * 1e-3 / NUM_ITER);
  }

  std::free(A);
  platform.finalize();
  return 0;
}
