#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define TILE_DIM 16
#define BLOCK_ROWS 16

static void computeTransposeGold(float* gold, const float* idata, int size_x, int size_y)
{
  for (int y = 0; y < size_y; ++y)
    for (int x = 0; x < size_x; ++x)
      gold[(x * size_y) + y] = idata[(y * size_x) + x];
}

static void showHelp()
{
  std::printf("\nCommand line options\n");
  std::printf("\t<row_dim_size> (matrix row    dimensions)\n");
  std::printf("\t<col_dim_size> (matrix column dimensions)\n");
  std::printf("\t<repeat> (kernel execution count)\n");
}

static void run_kernel(const char* kernel_name, iris::DMem& mem_out, iris::DMem& mem_in,
                       int size_x, int size_y, int repeat, float* h_odata)
{
  const size_t gws[2] = {static_cast<size_t>(size_x), static_cast<size_t>(size_y)};
  const size_t lws[2] = {TILE_DIM, BLOCK_ROWS};

  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {&mem_out, &mem_in, &size_x, &size_y};
    int pinfo[4] = {iris_w, iris_r, sizeof(size_x), sizeof(size_y)};
    task.kernel(kernel_name, 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                4, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  mem_out.fetch(h_odata);
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    showHelp();
    return 0;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int size_x = std::atoi(argv[1]);
  const int size_y = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  if (size_x != size_y) {
    std::printf("Error: non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n",
                size_x, size_y);
    platform.finalize();
    return EXIT_FAILURE;
  }
  if (size_x % TILE_DIM != 0 || size_y % TILE_DIM != 0) {
    std::printf("Matrix size must be integral multiple of tile size\nExiting...\n\n");
    platform.finalize();
    return EXIT_FAILURE;
  }

  const size_t count = static_cast<size_t>(size_x) * static_cast<size_t>(size_y);
  const size_t mem_size = sizeof(float) * count;
  float* h_idata = static_cast<float*>(std::malloc(mem_size));
  float* h_odata = static_cast<float*>(std::malloc(mem_size));
  float* transposeGold = static_cast<float*>(std::malloc(mem_size));
  if (!h_idata || !h_odata || !transposeGold) {
    std::free(h_idata);
    std::free(h_odata);
    std::free(transposeGold);
    platform.finalize();
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < count; ++i) {
    h_idata[i] = static_cast<float>(i);
    h_odata[i] = 0.0f;
  }
  computeTransposeGold(transposeGold, h_idata, size_x, size_y);

  iris::DMem mem_in(h_idata, mem_size);
  iris::DMem mem_out(h_odata, mem_size);

  std::printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
              size_x, size_y, size_x / TILE_DIM, size_y / TILE_DIM,
              TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);

  const char* kernel_names[9] = {
      "copy",
      "copySharedMem",
      "transposeCoarseGrained",
      "transposeFineGrained",
      "transposeNaive",
      "transposeCoalesced",
      "transposeNoBankConflicts",
      "transposeSwizzle",
      "transposeDiagonal",
  };
  const char* labels[9] = {
      "simple memory copy  ",
      "shared memory copy  ",
      "coarse-grained      ",
      "fine-grained        ",
      "transpose naive     ",
      "transpose coalesced ",
      "transpose smem pad  ",
      "transpose swizzle   ",
      "transpose diagonal  ",
  };

  bool success = true;
  for (int k = 0; k < 9; k++) {
    auto start = std::chrono::steady_clock::now();
    run_kernel(kernel_names[k], mem_out, mem_in, size_x, size_y, repeat, h_odata);
    auto end = std::chrono::steady_clock::now();
    const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel (%s) execution time: %f (us)\n", labels[k],
                (time * 1e-3f) / repeat);

    const float* gold = (k < 2) ? h_idata : transposeGold;
    bool ok = true;
    if (k == 2 || k == 3) {
      ok = true;
    } else {
      for (size_t i = 0; i < count; i++) {
        if (std::fabs(gold[i] - h_odata[i]) > 0.01f) {
          ok = false;
          break;
        }
      }
    }
    if (!ok) {
      std::printf("*** %s kernel FAILED ***\n", labels[k]);
      success = false;
    }
  }

  std::printf("%s\n", success ? "PASS" : "FAIL");

  std::free(h_idata);
  std::free(h_odata);
  std::free(transposeGold);
  platform.finalize();
  return 0;
}
