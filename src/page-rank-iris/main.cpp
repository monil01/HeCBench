#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#include "reference.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

const int max_iter = 1000;
const float threshold = 1e-16f;

static int *random_pages(int n, unsigned int *noutlinks, int divisor)
{
  int *pages = (int*)std::malloc((size_t)n * n * sizeof(int));
  if (divisor <= 0) {
    std::fprintf(stderr, "ERROR: Invalid divisor '%d' for random initialization, divisor should be greater or equal to 1\n", divisor);
    std::exit(1);
  }

  for (int i = 0; i < n; ++i) {
    noutlinks[i] = 0;
    for (int j = 0; j < n; ++j) {
      if (i != j && (std::abs(std::rand()) % divisor == 0)) {
        pages[(size_t)i * n + j] = 1;
        noutlinks[i] += 1;
      }
    }

    if (noutlinks[i] == 0) {
      int k;
      do {
        k = std::abs(std::rand()) % n;
      } while (k == i);
      pages[(size_t)i * n + k] = 1;
      noutlinks[i] = 1;
    }
  }
  return pages;
}

static void init_array(float *a, int n, float val)
{
  for (int i = 0; i < n; ++i) {
    a[i] = val;
  }
}

static void usage(char *argv[])
{
  std::fprintf(stderr, "Usage: %s [-n number of pages] [-i max iterations] [-t threshold] [-q divisor for zero density]\n", argv[0]);
}

static struct option size_opts[] =
{
  {"number of pages", 1, NULL, 'n'},
  {"max number of iterations", 1, NULL, 'i'},
  {"minimum threshold", 1, NULL, 't'},
  {"divisor for zero density", 1, NULL, 'q'},
  {0, 0, 0, 0}
};

static float maximum_dif(float *difs, int n)
{
  float max = 0.0f;
  for (int i = 0; i < n; ++i) {
    max = difs[i] > max ? difs[i] : max;
  }
  return max;
}

int main(int argc, char *argv[])
{
  int n = 1000;
  int iter = max_iter;
  float thresh = threshold;
  int divisor = 2;

  int opt;
  int opt_index = 0;
  while ((opt = getopt_long(argc, argv, "::n:i:t:q:", size_opts, &opt_index)) != -1) {
    switch (opt) {
      case 'n': n = std::atoi(optarg); break;
      case 'i': iter = std::atoi(optarg); break;
      case 't': thresh = std::atof(optarg); break;
      case 'q': divisor = std::atoi(optarg); break;
      default:
        usage(argv);
        return EXIT_FAILURE;
    }
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  size_t rank_size = (size_t)n * sizeof(float);
  size_t map_size = (size_t)n * n * sizeof(float);
  size_t page_size = (size_t)n * n * sizeof(int);
  size_t link_size = (size_t)n * sizeof(unsigned int);

  float *page_ranks = (float*)std::malloc(rank_size);
  float *maps = (float*)std::malloc(map_size);
  unsigned int *noutlinks = (unsigned int*)std::malloc(link_size);
  float *diffs = (float*)std::malloc(rank_size);
  if (!page_ranks || !maps || !noutlinks || !diffs) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(page_ranks);
    std::free(maps);
    std::free(noutlinks);
    std::free(diffs);
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < n; ++i) {
    noutlinks[i] = 0;
  }
  int *pages = random_pages(n, noutlinks, divisor);
  init_array(page_ranks, n, 1.0f / (float)n);
  std::memset(diffs, 0, rank_size);

  iris::DMem mem_pages(pages, page_size);
  iris::DMem mem_page_ranks(page_ranks, rank_size);
  iris::DMem mem_maps(maps, map_size);
  iris::DMem mem_noutlinks(noutlinks, link_size);
  iris::DMem mem_diffs(diffs, rank_size);

  size_t block_size = n < BLOCK_SIZE ? n : BLOCK_SIZE;
  size_t num_blocks = (n + block_size - 1) / block_size;
  size_t gws[1] = {num_blocks * block_size};
  size_t lws[1] = {block_size};

  int t;
  float max_diff = 99.0f;
  float max_diff_ref = 99.0f;
  double ktime = 0.0;
  std::printf("Threshold: %f\n", thresh);

  for (t = 1; t <= iter && max_diff >= thresh; ++t) {
    auto start = std::chrono::high_resolution_clock::now();

    iris::Task map_task;
    void *map_params[5] = {&mem_pages, &mem_page_ranks, &mem_maps, &mem_noutlinks, &n};
    int map_pinfo[5] = {iris_r, iris_r, iris_w, iris_r, sizeof(n)};
    map_task.kernel("map", 1, nullptr, gws, lws, 5, map_params, map_pinfo);
    map_task.submit(iris_gpu, nullptr, 1);

    iris::Task reduce_task;
    void *reduce_params[4] = {&mem_page_ranks, &mem_maps, &n, &mem_diffs};
    int reduce_pinfo[4] = {iris_rw, iris_r, sizeof(n), iris_rw};
    reduce_task.kernel("reduce", 1, nullptr, gws, lws, 4, reduce_params, reduce_pinfo);
    reduce_task.flush_out(mem_diffs);
    reduce_task.submit(iris_gpu, nullptr, 1);

    auto end = std::chrono::high_resolution_clock::now();
    ktime += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    mem_diffs.fetch(diffs);
    max_diff = maximum_dif(diffs, n);
  }

  std::fprintf(stderr, "Max difference %f is reached at iteration %d\n", max_diff, t);
  std::printf("\"Options\": \"-n %d -i %d -t %f\". Total kernel execution time: %lf (s)\n",
              n, iter, thresh, ktime);

  std::memset(diffs, 0, rank_size);
  for (t = 1; t <= iter && max_diff_ref >= thresh; ++t) {
    map_ref(pages, page_ranks, maps, noutlinks, n);
    reduce_ref(page_ranks, maps, n, diffs);
    max_diff_ref = maximum_dif(diffs, n);
  }
  bool ok = std::fabs(max_diff - max_diff_ref) < 1e-3f;
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(pages);
  std::free(maps);
  std::free(page_ranks);
  std::free(noutlinks);
  std::free(diffs);
  platform.finalize();
  return ok ? 0 : 1;
}
