#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "reference.h"

#define BLOCK_SIZE 256

static double LCG_random_double(uint64_t* seed)
{
  const unsigned long m = 9223372036854775808ULL;
  const unsigned long a = 2806196910506780709ULL;
  const unsigned long c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return static_cast<double>(*seed) / static_cast<double>(m);
}

static void run_damage_kernel(
    const char* name,
    int n,
    int m,
    int repeat,
    iris::DMem& mem_nlist,
    iris::DMem& mem_family,
    iris::DMem& mem_n_neigh,
    iris::DMem& mem_damage)
{
  const size_t gws[1] = { static_cast<size_t>(m) * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[5] = { &n, &mem_nlist, &mem_family, &mem_n_neigh, &mem_damage };
    int pinfo[5] = { sizeof(n), iris_r, iris_r, iris_w, iris_w };
    task.kernel(name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
    task.flush_out(mem_n_neigh);
    task.flush_out(mem_damage);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  const int m = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int padded_n = m * BLOCK_SIZE;

  int* nlist = static_cast<int*>(std::malloc(sizeof(int) * padded_n));
  int* family = static_cast<int*>(std::malloc(sizeof(int) * m));
  int* n_neigh = static_cast<int*>(std::malloc(sizeof(int) * m));
  double* damage = static_cast<double*>(std::malloc(sizeof(double) * m));
  if (!nlist || !family || !n_neigh || !damage) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }

  uint64_t seed = 123;
  for (int i = 0; i < n; i++) {
    nlist[i] = (LCG_random_double(&seed) > 0.5) ? 1 : -1;
  }
  for (int i = n; i < padded_n; i++) {
    nlist[i] = -1;
  }

  for (int i = 0; i < m; i++) {
    int s = 0;
    for (int j = 0; j < BLOCK_SIZE; j++) {
      s += (nlist[i * BLOCK_SIZE + j] != -1) ? 1 : 0;
    }
    family[i] = s + 1 + s * LCG_random_double(&seed);
  }

  iris::DMem mem_nlist(nlist, sizeof(int) * n);
  iris::DMem mem_family(family, sizeof(int) * m);
  iris::DMem mem_n_neigh(n_neigh, sizeof(int) * m);
  iris::DMem mem_damage(damage, sizeof(double) * m);

  int errors = 0;
  run_damage_kernel("damage_of_node", n, m, repeat, mem_nlist, mem_family, mem_n_neigh, mem_damage);
  mem_n_neigh.fetch(n_neigh);
  mem_damage.fetch(damage);
  errors += validate(BLOCK_SIZE, m, n, nlist, family, n_neigh, damage) ? 0 : 1;

  run_damage_kernel("damage_of_node_optimized", n, m, repeat, mem_nlist, mem_family, mem_n_neigh, mem_damage);
  mem_n_neigh.fetch(n_neigh);
  mem_damage.fetch(damage);
  errors += validate(BLOCK_SIZE, m, n, nlist, family, n_neigh, damage) ? 0 : 1;

  std::free(nlist);
  std::free(family);
  std::free(n_neigh);
  std::free(damage);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}
