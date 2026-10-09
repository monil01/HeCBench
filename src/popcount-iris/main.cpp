#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BLOCK_SIZE 256

static int popcount_ref(unsigned long x)
{
  int count;
  for (count = 0; x; count++) {
    x &= x - 1;
  }
  return count;
}

static bool checkResults(const unsigned long* d, const int* r, const int length)
{
  int error = 0;
  for (int i = 0; i < length; i++) {
    if (popcount_ref(d[i]) != r[i]) {
      error = 1;
      break;
    }
  }

  std::printf("%s\n", error ? "Fail" : "Success");
  return error == 0;
}

static double run_popcount_kernel(const char* kernel_name,
                                  int length,
                                  int repeat,
                                  iris::DMem& mem_data,
                                  iris::DMem& mem_result)
{
  const size_t blocks = (static_cast<size_t>(length) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[3] = { &mem_data, &mem_result, &length };
    int pinfo[3] = { iris_r, iris_w, sizeof(length) };
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 3, params, pinfo);
    if (n == repeat - 1) {
      task.flush_out(mem_result);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-3) / repeat;
}

static double run_popcount_lut_kernel(int length,
                                      int repeat,
                                      iris::DMem& mem_data,
                                      iris::DMem& mem_result,
                                      iris::DMem& mem_lut)
{
  const size_t blocks = (static_cast<size_t>(length) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[4] = { &mem_data, &mem_result, &length, &mem_lut };
    int pinfo[4] = { iris_r, iris_w, sizeof(length), iris_r };
    task.kernel("pc5", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
    if (n == repeat - 1) {
      task.flush_out(mem_result);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-3) / repeat;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <length> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int length = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  std::vector<unsigned long> data(length);
  std::vector<int> result(length);

  std::srand(2);
  for (int i = 0; i < length; i++) {
    unsigned long t = static_cast<unsigned long>(std::rand()) << 32;
    data[i] = t | static_cast<unsigned long>(std::rand());
  }

  const unsigned char lut[256] = {
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
  };

  iris::DMem mem_data(data.data(), sizeof(unsigned long) * static_cast<size_t>(length));
  iris::DMem mem_result(result.data(), sizeof(int) * static_cast<size_t>(length));
  iris::DMem mem_lut(const_cast<unsigned char*>(lut), sizeof(lut));

  bool all_ok = true;

  double us = run_popcount_kernel("pc1", length, repeat, mem_data, mem_result);
  std::printf("Average kernel execution time (pc1): %f (us)\n", us);
  mem_result.fetch(result.data());
  all_ok = checkResults(data.data(), result.data(), length) && all_ok;

  us = run_popcount_kernel("pc2", length, repeat, mem_data, mem_result);
  std::printf("Average kernel execution time (pc2): %f (us)\n", us);
  mem_result.fetch(result.data());
  all_ok = checkResults(data.data(), result.data(), length) && all_ok;

  us = run_popcount_kernel("pc3", length, repeat, mem_data, mem_result);
  std::printf("Average kernel execution time (pc3): %f (us)\n", us);
  mem_result.fetch(result.data());
  all_ok = checkResults(data.data(), result.data(), length) && all_ok;

  us = run_popcount_kernel("pc4", length, repeat, mem_data, mem_result);
  std::printf("Average kernel execution time (pc4): %f (us)\n", us);
  mem_result.fetch(result.data());
  all_ok = checkResults(data.data(), result.data(), length) && all_ok;

  us = run_popcount_lut_kernel(length, repeat, mem_data, mem_result, mem_lut);
  std::printf("Average kernel execution time (pc5): %f (us)\n", us);
  mem_result.fetch(result.data());
  all_ok = checkResults(data.data(), result.data(), length) && all_ok;

  us = run_popcount_kernel("pc6", length, repeat, mem_data, mem_result);
  std::printf("Average kernel execution time (pc6): %f (us)\n", us);
  mem_result.fetch(result.data());
  all_ok = checkResults(data.data(), result.data(), length) && all_ok;

  platform.finalize();
  return all_ok ? 0 : 1;
}
