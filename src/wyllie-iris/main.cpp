#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "utils.h"

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: ./%s <list size> <0 or 1> <repeat>", argv[0]);
    std::printf("0 and 1 indicate an ordered list and a random list, respectively\n");
    return -1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int elems = std::atoi(argv[1]);
  const int setRandomList = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  std::vector<int> next(elems);
  std::vector<int> rank(elems);
  std::vector<long> list(elems);
  std::vector<long> d_res(elems);
  std::vector<long> h_res(elems);

  if (setRandomList) {
    random_list(next);
  } else {
    ordered_list(next);
  }

  for (int i = 0; i < elems; i++) {
    rank[i] = next[i] == NIL ? 0 : 1;
  }

  for (int i = 0; i < elems; i++) {
    list[i] = (static_cast<long>(next[i]) << 32) | rank[i];
  }

  iris::DMem mem_list(list.data(), sizeof(long) * static_cast<size_t>(elems));
  const size_t block = 256;
  size_t gws[1] = { ((static_cast<size_t>(elems) + block - 1) / block) * block };
  size_t lws[1] = { block };

  double time = 0.0;
  for (int i = 0; i <= repeat; i++) {
    mem_list.update(list.data());

    auto start = std::chrono::steady_clock::now();
    iris::Task task;
    void* params[2] = { &mem_list, const_cast<int*>(&elems) };
    int pinfo[2] = { iris_rw, sizeof(elems) };
    task.kernel("wyllie", 1, nullptr, gws, lws, 2, params, pinfo);
    if (i == repeat) task.flush_out(mem_list);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();

    if (i > 0) time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  std::printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  mem_list.fetch(d_res.data());

  for (int i = 0; i < elems; i++) d_res[i] &= MASK;

  h_res[0] = elems - 1;
  int i = 0;
  for (int r = 1; r < elems; r++) {
    h_res[next[i]] = elems - 1 - r;
    i = next[i];
  }

  std::printf("%s\n", (h_res == d_res) ? "PASS" : "FAIL");

  platform.finalize();
  return (h_res == d_res) ? 0 : 1;
}
