#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct float2 {
  float x;
  float y;
};

static constexpr int VERTICES = 600;
static constexpr int BLOCK_SIZE_X = 256;

static size_t round_up(size_t value, size_t block) {
  return ((value + block - 1) / block) * block;
}

static void launch_pnpoly_base(iris::DMem& bitmap, iris::DMem& point,
                               iris::DMem& vertex, int n, int repeat) {
  const size_t gws[] = {round_up(static_cast<size_t>(n), BLOCK_SIZE_X)};
  const size_t lws[] = {BLOCK_SIZE_X};
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[] = {&bitmap, &point, &vertex, &n};
    int pinfo[] = {iris_w, iris_r, iris_r, sizeof(n)};
    task.kernel("pnpoly_base", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 4, params, pinfo);
    if (i + 1 == repeat) task.flush_out(bitmap);
    task.submit(iris_gpu, nullptr, 1);
  }
}

static void launch_pnpoly_opt(const char* symbol, int tile_size,
                              iris::DMem& bitmap, iris::DMem& point,
                              iris::DMem& vertex, int n, int repeat,
                              bool flush) {
  const size_t tile_span = static_cast<size_t>(BLOCK_SIZE_X) * tile_size;
  const size_t blocks = (static_cast<size_t>(n) + tile_span - 1) / tile_span;
  const size_t gws[] = {blocks * BLOCK_SIZE_X};
  const size_t lws[] = {BLOCK_SIZE_X};
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[] = {&bitmap, &point, &vertex, &n};
    int pinfo[] = {iris_w, iris_r, iris_r, sizeof(n)};
    task.kernel(symbol, 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 4, params, pinfo);
    if (flush && i + 1 == repeat) task.flush_out(bitmap);
    task.submit(iris_gpu, nullptr, 1);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int nPoints = 20000000;
  const int vertices = VERTICES;

  std::default_random_engine rng(123);
  std::normal_distribution<float> distribution(0, 1);

  std::vector<float2> point(nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  std::vector<float2> vertex(vertices);
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  std::vector<int> bitmap_ref(nPoints);
  std::vector<int> bitmap_opt(nPoints);

  iris::DMem point_mem(point.data(), point.size() * sizeof(float2));
  iris::DMem vertex_mem(vertex.data(), vertex.size() * sizeof(float2));
  iris::DMem bitmap_ref_mem(bitmap_ref.data(), bitmap_ref.size() * sizeof(int));
  iris::DMem bitmap_opt_mem(bitmap_opt.data(), bitmap_opt.size() * sizeof(int));

  auto start = std::chrono::steady_clock::now();
  launch_pnpoly_base(bitmap_ref_mem, point_mem, vertex_mem, nPoints, repeat);
  bitmap_ref_mem.fetch(bitmap_ref.data());
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time (pnpoly_base): %f (s)\n",
              (time * 1e-9f) / repeat);

  struct OptKernel {
    const char* symbol;
    int tile;
    const char* label;
  };
  const OptKernel kernels[] = {
      {"pnpoly_opt_1", 1, "pnpoly_opt<1>"},
      {"pnpoly_opt_2", 2, "pnpoly_opt<2>"},
      {"pnpoly_opt_4", 4, "pnpoly_opt<4>"},
      {"pnpoly_opt_8", 8, "pnpoly_opt<8>"},
      {"pnpoly_opt_16", 16, "pnpoly_opt<16>"},
      {"pnpoly_opt_32", 32, "pnpoly_opt<32>"},
      {"pnpoly_opt_64", 64, "pnpoly_opt<64>"},
  };

  bool all_ok = true;
  for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); ++i) {
    start = std::chrono::steady_clock::now();
    launch_pnpoly_opt(kernels[i].symbol, kernels[i].tile, bitmap_opt_mem,
                      point_mem, vertex_mem, nPoints, repeat,
                      true);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    std::printf("Average kernel execution time (%s): %f (s)\n",
                kernels[i].label, (time * 1e-9f) / repeat);
    bool matches = std::memcmp(bitmap_opt.data(), bitmap_ref.data(),
                                bitmap_ref.size() * sizeof(int)) == 0;
    all_ok = matches && all_ok;
    std::printf("Variant %s: %s\n", kernels[i].label, matches ? "PASS" : "FAIL");
  }

  bitmap_opt_mem.fetch(bitmap_opt.data());
  int error = std::memcmp(bitmap_opt.data(), bitmap_ref.data(),
                          bitmap_ref.size() * sizeof(int));
  int checksum = 0;
  for (int v : bitmap_opt) checksum += v;
  std::printf("Checksum: %d\n", checksum);
  std::printf("%s\n", (error || !all_ok) ? "FAIL" : "PASS");

  platform.finalize();
  return (error || !all_ok) ? 1 : 0;
}
