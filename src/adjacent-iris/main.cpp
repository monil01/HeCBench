#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void Initialize(int* h_in, int num_items)
{
  for (int i = 0; i < num_items; ++i) {
    h_in[i] = i % 17;
  }
}

static void SubmitAdjacent(iris::DMem& in_mem, iris::DMem& out_mem, int grid_size,
                           int block_threads, bool subtract_left, bool flush)
{
  int subtract_left_arg = subtract_left ? 1 : 0;
  size_t gws[1] = {static_cast<size_t>(grid_size) * static_cast<size_t>(block_threads)};
  size_t lws[1] = {static_cast<size_t>(block_threads)};
  iris::Task task;
  void* params[3] = {&in_mem, &out_mem, &subtract_left_arg};
  int pinfo[3] = {iris_r, iris_w, sizeof(subtract_left_arg)};
  task.kernel("adjacent_diff", 1, nullptr, gws, lws, 3, params, pinfo);
  if (flush) task.flush_out(out_mem);
  task.submit(iris_gpu, nullptr, 1);
}

static void SubmitAdjacentInPlace(iris::DMem& mem, int grid_size, int block_threads,
                                  bool subtract_left)
{
  int subtract_left_arg = subtract_left ? 1 : 0;
  size_t gws[1] = {static_cast<size_t>(grid_size) * static_cast<size_t>(block_threads)};
  size_t lws[1] = {static_cast<size_t>(block_threads)};
  iris::Task task;
  void* params[3] = {&mem, &mem, &subtract_left_arg};
  int pinfo[3] = {iris_rw, iris_rw, sizeof(subtract_left_arg)};
  task.kernel("adjacent_diff", 1, nullptr, gws, lws, 3, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

template <int BLOCK_THREADS>
static bool Test(int requested_items, int repeat)
{
  const int ITEMS_PER_THREAD = 4;
  const int items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  int num_items = (requested_items + items_per_block - 1) / items_per_block * items_per_block;

  int* h_in = new int[num_items];
  int* h_out = new int[num_items];
  int* r_out = new int[num_items];

  Initialize(h_in, num_items);
  std::memset(h_out, 0, sizeof(int) * static_cast<size_t>(num_items));
  std::memset(r_out, 0, sizeof(int) * static_cast<size_t>(num_items));

  iris::DMem in_mem(h_in, sizeof(int) * static_cast<size_t>(num_items));
  iris::DMem out_mem(h_out, sizeof(int) * static_cast<size_t>(num_items));
  int grid_size = num_items / items_per_block;

  in_mem.update(h_in);
  for (int i = 0; i < repeat; i++) {
    SubmitAdjacent(in_mem, out_mem, grid_size, BLOCK_THREADS, true, i == repeat - 1);
  }

  for (int b = 0; b < grid_size; b++) {
    auto in = h_in + b * items_per_block;
    auto out = r_out + b * items_per_block;
    for (int i = 0; i < items_per_block; i++) {
      out[i] = (i - 1) < 0 ? in[i] : in[i] - in[i - 1];
    }
  }

  int compare = std::memcmp(r_out, h_out, sizeof(int) * static_cast<size_t>(num_items));
  std::printf("%s\n", compare ? "FAIL" : "PASS");
  bool ok = compare == 0;

  in_mem.update(h_in);
  for (int i = 0; i < repeat; i++) {
    SubmitAdjacent(in_mem, out_mem, grid_size, BLOCK_THREADS, false, i == repeat - 1);
  }

  for (int b = 0; b < grid_size; b++) {
    auto in = h_in + b * items_per_block;
    auto out = r_out + b * items_per_block;
    for (int i = 0; i < items_per_block; i++) {
      out[i] = (i + 1) >= items_per_block ? in[i] : in[i] - in[i + 1];
    }
  }
  compare = std::memcmp(r_out, h_out, sizeof(int) * static_cast<size_t>(num_items));
  std::printf("%s\n", compare ? "FAIL" : "PASS");
  ok = ok && compare == 0;

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    SubmitAdjacent(in_mem, out_mem, grid_size, BLOCK_THREADS, true, false);
    SubmitAdjacentInPlace(out_mem, grid_size, BLOCK_THREADS, false);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the kernels (thread block size = %4d): %f (us)\n",
              BLOCK_THREADS, (time * 1e-3f) / repeat);

  delete[] h_in;
  delete[] h_out;
  delete[] r_out;
  return ok;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (nelems <= 0 || repeat <= 0) {
    std::fprintf(stderr, "number of elements and repeat must be positive\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  bool ok = true;
  ok = Test<64>(nelems, repeat) && ok;
  ok = Test<128>(nelems, repeat) && ok;
  ok = Test<256>(nelems, repeat) && ok;
  ok = Test<512>(nelems, repeat) && ok;
  ok = Test<1024>(nelems, repeat) && ok;

  platform.finalize();
  return ok ? 0 : 1;
}
