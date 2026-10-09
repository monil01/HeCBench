#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool verify(const unsigned int* sorted_keys, const unsigned int* keys,
                   const unsigned int threads, const int n)
{
  unsigned int m1[16], m2[16];
  const int block_items = static_cast<int>(threads) * 4;

  for (int i = 0; i < n; i += block_items) {
    for (int j = 0; j < block_items - 1; j++) {
      if (sorted_keys[i + j] > sorted_keys[i + j + 1]) return false;
    }
  }

  for (int i = 0; i < n; i++) {
    if (sorted_keys[i] >= 16) return false;
  }

  for (int i = 0; i < n; i += block_items) {
    std::memset(m1, 0, sizeof(m1));
    std::memset(m2, 0, sizeof(m2));
    for (int j = 0; j < block_items; j++) {
      m1[keys[i + j]]++;
      m2[sorted_keys[i + j]]++;
    }
    if (std::memcmp(m1, m2, sizeof(m1)) != 0) return false;
  }
  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (n <= 0 || repeat <= 0 || (n % 512) != 0) {
    std::printf("Usage: %s <number of keys multiple of 512> <repeat>\n", argv[0]);
    platform.finalize();
    return 1;
  }

  const size_t bytes = static_cast<size_t>(n) * sizeof(unsigned int);
  unsigned int* keys = static_cast<unsigned int*>(std::malloc(bytes));
  unsigned int* out = static_cast<unsigned int*>(std::malloc(bytes));
  if (!keys || !out) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(out);
    std::free(keys);
    platform.finalize();
    return 1;
  }

  std::srand(512);
  for (int i = 0; i < n; i++) keys[i] = static_cast<unsigned int>(std::rand() % 16);
  std::memset(out, 0, bytes);

  const unsigned int startbit = 0;
  const unsigned int nbits = 4;
  const unsigned int threads = 128;
  const size_t teams = static_cast<size_t>(n) / 4 / threads;
  const size_t gws[1] = { teams * threads };
  const size_t lws[1] = { threads };

  iris::DMem mem_keys(keys, bytes);
  iris::DMem mem_out(out, bytes);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = { &mem_keys, &mem_out, const_cast<unsigned int*>(&nbits),
                        const_cast<unsigned int*>(&startbit) };
    int pinfo[4] = { iris_r, iris_w, sizeof(nbits), sizeof(startbit) };
    task.kernel("split_radix_sort_blocks_keys", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 4, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  bool check = verify(out, keys, threads, n);
  std::printf("%s\n", check ? "PASS" : "FAIL");

  std::free(out);
  std::free(keys);
  platform.finalize();
  return check ? 0 : 1;
}
