#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static inline bool valid(int x)
{
  return x > 0;
}

template <int N>
static bool bscan(const int repeat, int warp_size)
{
  if (N < warp_size) {
    std::printf("Block size = %d, skipped (< warpSize %d)\n", N, warp_size);
    return true;
  }

  int h_in[N];
  int h_out[N];
  int ref_out[N];

  iris::DMem mem_in(h_in, N * sizeof(int));
  iris::DMem mem_out(h_out, N * sizeof(int));

  bool ok = true;
  double time = 0.0;
  std::srand(123);

  const size_t grid_size = 12 * 7 * 8 * 9 * 10;
  const size_t gws[1] = {grid_size * N};
  const size_t lws[1] = {N};

  int valid_count = 0;

  for (int iter = 0; iter < repeat; iter++) {
    for (int n = 0; n < N; n++) {
      h_in[n] = std::rand() % N - N / 2;
      if (valid(h_in[n])) valid_count++;
    }
    mem_in.update(h_in);

    auto start = std::chrono::steady_clock::now();
    iris::Task task;
    void* params[2] = {&mem_out, &mem_in};
    int pinfo[2] = {iris_w, iris_r};
    task.kernel("binary_scan", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 2, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    mem_out.fetch(h_out);

    ref_out[0] = 0;
    ok &= (h_out[0] == ref_out[0]);
    for (int i = 1; i < N; i++) {
      ref_out[i] = ref_out[i - 1] + (h_in[i - 1] > 0);
      ok &= (ref_out[i] == h_out[i]);
    }
    if (!ok) break;
  }

  std::printf("Block size = %d, ratio of valid elements = %f, verify = %s\n",
              N, valid_count * 1.f / (N * repeat), ok ? "PASS" : "FAIL");

  if (ok) {
    std::printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);
    std::printf("Billion elements per second: %f\n\n", grid_size * N * repeat / time);
  }
  return ok;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int warp_size = 64;

  bool ok = true;
  ok &= bscan<32>(repeat, warp_size);
  ok &= bscan<64>(repeat, warp_size);
  ok &= bscan<128>(repeat, warp_size);
  ok &= bscan<256>(repeat, warp_size);
  ok &= bscan<512>(repeat, warp_size);
  ok &= bscan<1024>(repeat, warp_size);

  platform.finalize();
  return ok ? 0 : 1;
}
