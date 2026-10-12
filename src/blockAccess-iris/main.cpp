#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

static void submit_kernel(
    const char* name,
    iris::DMem& mem_a,
    iris::DMem& mem_out,
    size_t n,
    size_t* gws,
    size_t* lws,
    bool flush)
{
  iris::Task task;
  void* params[3] = {&mem_a, &mem_out, &n};
  int pinfo[3] = {iris_r, iris_w, sizeof(n)};
  task.kernel(name, 1, nullptr, gws, lws, 3, params, pinfo);
  if (flush) task.flush_out(mem_out);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Block access N elements where N is represented as rows x columns\n");
    std::printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nrows = std::atoi(argv[1]);
  const int ncols = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  const size_t n = ((size_t)nrows * ncols + 3) / 4 * 4;
  const size_t A_size = n * sizeof(float);
  const size_t out_size = n * sizeof(unsigned char);

  float *A = static_cast<float*>(std::malloc(A_size));
  unsigned char *out = static_cast<unsigned char*>(std::malloc(out_size));
  unsigned char *out_ref = static_cast<unsigned char*>(std::malloc(out_size));
  if (!A || !out || !out_ref) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(A);
    std::free(out);
    std::free(out_ref);
    platform.finalize();
    return 1;
  }

  std::mt19937 gen{19937};
  std::normal_distribution<float> d{-128.0, 127.0};
  for (size_t i = 0; i < n; i++) {
    A[i] = d(gen);
    out[i] = 0;
    out_ref[i] = 0;
  }

  iris::DMem mem_a(A, A_size);
  iris::DMem mem_out(out, out_size);
  iris::DMem mem_out_ref(out_ref, out_size);

  const int block_size = 256;
  const int grid_blocks = 1024;
  size_t gws[1] = {static_cast<size_t>(grid_blocks * block_size)};
  size_t lws[1] = {block_size};

  submit_kernel("reference_kernel", mem_a, mem_out_ref, n, gws, lws, true);
  submit_kernel("block_access_kernel", mem_a, mem_out, n, gws, lws, true);
  mem_out_ref.fetch(out_ref);
  mem_out.fetch(out);

  bool error = false;
  for (size_t i = 0; i < n; i++) {
    unsigned char t = int(A[i]);
    if (out[i] != t) {
      std::printf("@%zu: out[%u] != %u\n", i, out[i], t);
      error = true;
      break;
    }
    if (out_ref[i] != t) {
      std::printf("@%zu: out_ref[%u] != %u\n", i, out_ref[i], t);
      error = true;
      break;
    }
  }
  std::printf("%s\n", error ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    submit_kernel("reference_kernel", mem_a, mem_out_ref, n, gws, lws, false);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the reference kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    submit_kernel("block_access_kernel", mem_a, mem_out, n, gws, lws, false);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the blockAccess kernel: %f (us)\n", (time * 1e-3f) / repeat);

  std::free(A);
  std::free(out);
  std::free(out_ref);
  platform.finalize();
  return error ? 2 : 0;
}
