#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: ./%s <iterations>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int iteration = std::atoi(argv[1]);
  const int len = 256;
  const int elem_size = len * static_cast<int>(sizeof(int));

  int test[len];
  int gold_odd[len];
  int gold_even[len];

  for (int i = 0; i < len; i++) {
    gold_odd[i] = len - i - 1;
    gold_even[i] = i;
    test[i] = i;
  }

  iris::DMem mem_test(test, elem_size);

  std::default_random_engine generator(123);
  std::uniform_int_distribution<int> distribution(100, 9999);

  long time = 0;
  int error = 0;

  for (int i = 0; i < iteration; i++) {
    const int count = distribution(generator);
    std::memcpy(test, gold_even, elem_size);
    mem_test.update(test);

    auto start = std::chrono::steady_clock::now();
    for (int j = 0; j < count; j++) {
      iris::Task task;
      void* params[2] = {&mem_test, const_cast<int*>(&len)};
      int pinfo[2] = {iris_rw, sizeof(len)};
      size_t gws[1] = {static_cast<size_t>(len)};
      size_t lws[1] = {static_cast<size_t>(len)};
      task.kernel("reverse", 1, nullptr, gws, lws, 2, params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
    }
    iris::Task flush;
    flush.flush_out(mem_test);
    flush.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    mem_test.fetch(test);
    if (count % 2 == 0) {
      error = std::memcmp(test, gold_even, elem_size);
    } else {
      error = std::memcmp(test, gold_odd, elem_size);
    }

    if (error) {
      break;
    }
  }

  std::printf("Total kernel execution time: %f (s)\n", time * 1e-9f);
  std::printf("%s\n", error ? "FAIL" : "PASS");

  platform.finalize();
  return error ? 1 : 0;
}
