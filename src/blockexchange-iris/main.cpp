#include <iris/iris.hpp>

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nrows = std::atoi(argv[1]);
  const int ncols = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  const size_t n_size = static_cast<size_t>(nrows) * ncols;
  if (n_size > static_cast<size_t>(INT32_MAX)) {
    std::fprintf(stderr, "element count exceeds int range\n");
    platform.finalize();
    return 1;
  }
  const int n = static_cast<int>(n_size);
  const size_t bytes = n_size * sizeof(int);

  int* input = static_cast<int*>(std::malloc(bytes));
  int* output = static_cast<int*>(std::malloc(bytes));
  if (!input || !output) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(input);
    std::free(output);
    platform.finalize();
    return 1;
  }

  for (size_t i = 0; i < n_size; i++) input[i] = static_cast<int>(i);

  iris::DMem mem_input(input, bytes);
  iris::DMem mem_output(output, bytes);

  constexpr int block_size = 256;
  const size_t blocks = (n_size + block_size - 1) / block_size;
  const size_t gws[1] = {blocks * block_size};
  const size_t lws[1] = {block_size};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = {&mem_input, &mem_output, const_cast<int*>(&n)};
    int pinfo[3] = {iris_r, iris_w, sizeof(n)};
    task.kernel("blockexchange_kernel", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  iris::Task flush;
  flush.flush_out(mem_output);
  flush.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of kernel: %f (us)\n", (time * 1e-3f) / repeat);

  mem_output.fetch(output);

  const int items_per_thread = 4;
  const int items_per_block = items_per_thread * block_size;
  if (n % items_per_block == 0) {
    bool ok = true;
    for (size_t k = 0; k < n_size; k += items_per_block) {
      size_t i = k;
      for (int j = 0; j < items_per_thread; j++) {
        for (int m = 0; m < block_size - 1; m++) {
          if (i + (m + 1) * items_per_thread < n_size &&
              output[i + (m + 1) * items_per_thread] - output[i + m * items_per_thread] != 1) {
            std::printf("Error at index %zu\n", i + (m + 1) * items_per_thread);
            ok = false;
            goto stop;
          }
        }
        i++;
      }
    }
  stop:
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) {
      std::free(input);
      std::free(output);
      platform.finalize();
      return 1;
    }
  }

  std::free(input);
  std::free(output);
  platform.finalize();
  return 0;
}
