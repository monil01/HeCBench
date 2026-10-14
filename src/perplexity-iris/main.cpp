#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

static void perplexity_search(
    iris::DMem& mem_distances,
    iris::DMem& mem_p,
    float perplexity,
    int epochs,
    float tol,
    int n,
    int dim,
    double& time)
{
  float desired_entropy = logf(perplexity);
  const size_t gws[1] = { static_cast<size_t>(((n + 255) / 256) * 256) };
  const size_t lws[1] = { 256 };

  auto start = std::chrono::steady_clock::now();
  iris::Task task;
  void* params[7] = { &mem_distances, &mem_p, &perplexity, &desired_entropy, &epochs, &tol, &n };
  int pinfo[7] = { iris_r, iris_w, sizeof(perplexity), sizeof(desired_entropy), sizeof(epochs), sizeof(tol), sizeof(n) };
  task.kernel("sigmas_kernel", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 7, params, pinfo);
  task.flush_out(mem_p);
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of points> <perplexity> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int n = std::atoi(argv[1]);
  const int p = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  const int n_nbrs = 4 * p;
  const int max_iter = 100;
  const float tol = 1e-8f;

  std::srand(123);
  std::vector<float> data(static_cast<size_t>(n) * n_nbrs);
  std::vector<float> h_data(static_cast<size_t>(n) * n_nbrs);
  std::vector<float> distance(static_cast<size_t>(n) * n_nbrs);
  for (int i = 0; i < n * n_nbrs; i++) {
    distance[i] = std::rand() / static_cast<float>(RAND_MAX);
  }

  iris::DMem mem_distance(distance.data(), sizeof(float) * distance.size());
  iris::DMem mem_data(data.data(), sizeof(float) * data.size());

  double time = 0.0;
  for (int i = 0; i < repeat; i++) {
    perplexity_search(mem_distance, mem_data, p, max_iter, tol, n, n_nbrs, time);
  }

  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  mem_data.fetch(data.data());

  reference(distance.data(), h_data.data(), p, max_iter, tol, n, n_nbrs);

  bool ok = true;
  for (int i = 0; i < n * n_nbrs; i++) {
    if (!std::isfinite(data[i]) || !std::isfinite(h_data[i]) || fabsf(data[i] - h_data[i]) > 1e-3f) {
      std::printf("%d %f %f\n", i, data[i], h_data[i]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  return ok ? 0 : 1;
}
