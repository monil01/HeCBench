#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int grid_size = 1;
static int repeat = 100;

static int initialize(int* in, int* ref, int n)
{
  int inclusive = 0;
  for (int i = 0; i < n; ++i) {
    in[i] = i % 17;
    ref[i] = inclusive;
    inclusive += in[i];
  }
  return inclusive;
}

static const char* algorithm_name(int algorithm)
{
  if (algorithm == 0) return "BLOCK_SCAN_RAKING";
  if (algorithm == 1) return "BLOCK_SCAN_RAKING_MEMOIZE";
  return "BLOCK_SCAN_WARP_SCANS";
}

static const char* kernel_name(int block_threads, int items_per_thread, int algorithm)
{
  if (algorithm == 0 && block_threads == 1024 && items_per_thread == 1) return "scan_raking_1024_1";
  if (algorithm == 0 && block_threads == 512 && items_per_thread == 2) return "scan_raking_512_2";
  if (algorithm == 0 && block_threads == 256 && items_per_thread == 4) return "scan_raking_256_4";
  if (algorithm == 0 && block_threads == 128 && items_per_thread == 8) return "scan_raking_128_8";
  if (algorithm == 0 && block_threads == 64 && items_per_thread == 16) return "scan_raking_64_16";
  if (algorithm == 1 && block_threads == 1024 && items_per_thread == 1) return "scan_memoize_1024_1";
  if (algorithm == 1 && block_threads == 512 && items_per_thread == 2) return "scan_memoize_512_2";
  if (algorithm == 1 && block_threads == 256 && items_per_thread == 4) return "scan_memoize_256_4";
  if (algorithm == 1 && block_threads == 128 && items_per_thread == 8) return "scan_memoize_128_8";
  if (algorithm == 1 && block_threads == 64 && items_per_thread == 16) return "scan_memoize_64_16";
  if (algorithm == 2 && block_threads == 1024 && items_per_thread == 1) return "scan_warp_1024_1";
  if (algorithm == 2 && block_threads == 512 && items_per_thread == 2) return "scan_warp_512_2";
  if (algorithm == 2 && block_threads == 256 && items_per_thread == 4) return "scan_warp_256_4";
  if (algorithm == 2 && block_threads == 128 && items_per_thread == 8) return "scan_warp_128_8";
  return "scan_warp_64_16";
}

static int run_test(int block_threads, int items_per_thread, int algorithm)
{
  const int tile_size = block_threads * items_per_thread;
  int* h_in = new int[tile_size];
  int* h_reference = new int[tile_size];
  int* h_out = new int[tile_size + 1];
  const int h_aggregate = initialize(h_in, h_reference, tile_size);
  std::memset(h_out, 0, sizeof(int) * (tile_size + 1));

  std::printf("BlockScan algorithm %s on %d items (%d timing iterations, %d blocks, %d threads, %d items per thread):\n",
              algorithm_name(algorithm), tile_size, repeat, grid_size, block_threads, items_per_thread);

  iris::DMem mem_in(h_in, sizeof(int) * tile_size);
  iris::DMem mem_out(h_out, sizeof(int) * (tile_size + 1));

  const size_t local_size = block_threads > 256 ? 256 : static_cast<size_t>(block_threads);
  const size_t gws[1] = { static_cast<size_t>(grid_size) * local_size };
  const size_t lws[1] = { local_size };
  const char* name = kernel_name(block_threads, items_per_thread, algorithm);

  {
    iris::Task task;
    void* params[2] = { &mem_in, &mem_out };
    int pinfo[2] = { iris_r, iris_w };
    task.kernel(name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 2, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }

  std::printf("\tOutput items: %s\n", std::memcmp(h_out, h_reference, sizeof(int) * tile_size) == 0 ? "PASS" : "FAIL");
  std::printf("\tAggregate: %s\n", h_out[tile_size] == h_aggregate ? "PASS" : "FAIL");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[2] = { &mem_in, &mem_out };
    int pinfo[2] = { iris_r, iris_w };
    task.kernel(name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 2, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  const float avg_millis = elapsed * 1e-6f / repeat;
  const float avg_items_per_sec = static_cast<float>(tile_size) * grid_size / avg_millis / 1000.0f;

  std::printf("\tAverage kernel time: %.4f ms\n", avg_millis);
  std::printf("\tAverage million items / sec: %.4f\n", avg_items_per_sec);

  const bool ok = std::memcmp(h_out, h_reference, sizeof(int) * tile_size) == 0 && h_out[tile_size] == h_aggregate;
  delete[] h_in;
  delete[] h_reference;
  delete[] h_out;
  return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("The benchmark evaluates the impacts of the number of threads ");
    std::printf("per block and the number of items per thread on the performance ");
    std::printf("of block-level scans\n");
    std::printf("Usage: %s <grid_size> <repeat>\n", argv[0]);
    std::printf("grid_size specifies the number of thread blocks");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  grid_size = std::atoi(argv[1]);
  repeat = std::atoi(argv[2]);

  int failures = 0;
  failures += run_test(1024, 1, 0);
  failures += run_test(512, 2, 0);
  failures += run_test(256, 4, 0);
  failures += run_test(128, 8, 0);
  failures += run_test(64, 16, 0);
  std::printf("-------------\n");
  failures += run_test(1024, 1, 1);
  failures += run_test(512, 2, 1);
  failures += run_test(256, 4, 1);
  failures += run_test(128, 8, 1);
  failures += run_test(64, 16, 1);
  std::printf("-------------\n");
  failures += run_test(1024, 1, 2);
  failures += run_test(512, 2, 2);
  failures += run_test(256, 4, 2);
  failures += run_test(128, 8, 2);
  failures += run_test(64, 16, 2);

  platform.finalize();
  return failures == 0 ? 0 : 1;
}
