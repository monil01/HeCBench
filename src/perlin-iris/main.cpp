#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static constexpr uint64_t WIN_WIDTH = 61440;
static constexpr uint64_t WIN_HEIGHT = 34560;

struct NoiseParams {
  float ppu;
  int seed;
  int octaves;
  float lacunarity;
  float persistence;
};

struct Dim3 {
  size_t x;
  size_t y;
  size_t z;
};

static std::ostream& operator<<(std::ostream& o, const Dim3& d)
{
  o << "(" << d.x << "," << d.y << "," << d.z << ")";
  return o;
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  NoiseParams params;
  params.ppu = 250.f;
  params.seed = 0;
  params.octaves = 3;
  params.lacunarity = 2.f;
  params.persistence = 0.5f;

  const size_t pixel_bytes = 4 * WIN_WIDTH * WIN_HEIGHT * sizeof(uint8_t);
  std::vector<uint8_t> h_pixels(pixel_bytes);
  iris::DMem d_pixels(h_pixels.data(), pixel_bytes);

  for (int n_streams = 1; n_streams <= 32; n_streams *= 2) {
    std::cout << std::endl;
    std::cout << "Using " << n_streams << " streams." << std::endl;

    const int partial_height = static_cast<int>(WIN_HEIGHT / n_streams);
    const Dim3 threads{16, 16, 1};
    const Dim3 blocks{(WIN_WIDTH + threads.x - 1) / threads.x,
                      (static_cast<size_t>(partial_height) + threads.y - 1) / threads.y,
                      1};
    const size_t total_threads = threads.x * blocks.x * threads.y * blocks.y;

    std::cout << "Each stream will calculate "
              << static_cast<uint64_t>(partial_height) * WIN_WIDTH << " pixels." << std::endl;
    std::cout << "For each stream, ";
    std::cout << "block size = " << threads << ", grid size = " << blocks
              << ", total threads = " << total_threads << std::endl;

    const size_t gws[2] = {blocks.x * threads.x, blocks.y * threads.y};
    const size_t lws[2] = {threads.x, threads.y};

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_streams; ++i) {
      int y_start = partial_height * i;
      iris::Task task;
      void* params_list[4] = {&y_start, const_cast<int*>(&partial_height), &params, &d_pixels};
      int pinfo[4] = {sizeof(y_start), sizeof(partial_height), sizeof(params), iris_rw};
      task.kernel("perlin", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                  4, params_list, pinfo);
      task.flush_out(d_pixels);
      task.submit(iris_gpu, nullptr, 1);
    }
    d_pixels.fetch(h_pixels.data());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Total kernel execution time " << time * 1e-6 << " (ms)" << std::endl;

    uint64_t checksum = 0;
    for (uint64_t i = 0; i < 4 * WIN_WIDTH * WIN_HEIGHT; ++i) checksum += h_pixels[i];
    std::cout << "checksum = " << checksum / (4 * WIN_WIDTH * WIN_HEIGHT) << std::endl;
  }

  platform.finalize();
  return 0;
}
