#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "reference.h"

static std::vector<char> mtf_iris(const std::vector<char>& word)
{
  std::vector<char> result(word.size());
  iris::DMem mem_word(const_cast<char*>(word.data()), word.size() * sizeof(char));
  iris::DMem mem_result(result.data(), result.size() * sizeof(char));
  mem_word.update(const_cast<char*>(word.data()));

  const unsigned long long len = static_cast<unsigned long long>(word.size());
  size_t gws[1] = { 1 };
  size_t lws[1] = { 1 };

  iris::Task task;
  void* params[3] = { &mem_word, &mem_result, const_cast<unsigned long long*>(&len) };
  int pinfo[3] = { iris_r, iris_w, sizeof(len) };
  task.kernel("mtf_kernel", 1, nullptr, gws, lws, 3, params, pinfo);
  task.flush_out(mem_result);
  task.submit(iris_gpu, nullptr, 1);

  mem_result.fetch(result.data());
  return result;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <string length> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const size_t len = std::atoll(argv[1]);
  const int repeat = std::atoi(argv[2]);
  const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::vector<char> word(len);
  std::srand(123);
  for (size_t i = 0; i < len; i++) word[i] = a[std::rand() % 52];

  auto d_result = mtf_iris(word);
  auto h_result = reference(word);
  bool ok = d_result == h_result;
  if (ok) {
    std::printf("PASS\n");
  } else {
    if (len < 16) {
      std::printf("host: ");
      for (size_t i = 0; i < len; i++) std::printf("%d ", h_result[i]);
      std::printf("\ndevice: ");
      for (size_t i = 0; i < len; i++) std::printf("%d ", d_result[i]);
      std::printf("\n");
    }
    std::printf("FAIL\n");
    platform.finalize();
    return 1;
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) mtf_iris(word);
  auto end = std::chrono::steady_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time: " << (time * 1e-9f) / repeat << " (s)\n";

  platform.finalize();
  return 0;
}
