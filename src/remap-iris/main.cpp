#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define NUM_THREADS 256

static void build_remap_orders(
    const std::vector<int>& input,
    std::vector<int>& first_order,
    std::vector<int>& second_order)
{
  const int n = static_cast<int>(input.size());
  first_order.resize(n);
  std::iota(first_order.begin(), first_order.end(), 0);
  std::sort(first_order.begin(), first_order.end(), [&](int a, int b) {
    if (input[a] != input[b]) return input[a] < input[b];
    return a < b;
  });

  second_order.clear();
  if (n == 0) return;
  second_order.push_back(0);
  for (int i = 1; i < n; ++i) {
    if (input[first_order[i]] != input[first_order[i - 1]]) {
      second_order.push_back(i);
    }
  }
}

static int eval_remap(int n, int repeat, int run)
{
  std::vector<int> input(static_cast<size_t>(n));
#ifdef EXAMPLE
  input[0] = 1; input[1] = 3; input[2] = 5;
  input[3] = 1; input[4] = 5; input[5] = 7;
  input[6] = 9;
#else
  std::srand(123);
  for (int i = 0; i < n; i++) {
    input[static_cast<size_t>(i)] = std::rand() % n;
  }
#endif

  std::vector<int> output(static_cast<size_t>(n));
  std::vector<int> first_order;
  std::vector<int> second_order;

  long seq_time = 0;
  long sort_time = 0;
  long unique_time = 0;
  long kernel_time = 0;
  long copy_time = 0;
  long alloc_time = 0;
  long dealloc_time = 0;

  auto offload_start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    auto start = std::chrono::steady_clock::now();
    first_order.resize(static_cast<size_t>(n));
    second_order.reserve(static_cast<size_t>(n));
    auto end = std::chrono::steady_clock::now();
    alloc_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    std::iota(first_order.begin(), first_order.end(), 0);
    end = std::chrono::steady_clock::now();
    seq_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    std::sort(first_order.begin(), first_order.end(), [&](int a, int b) {
      if (input[static_cast<size_t>(a)] != input[static_cast<size_t>(b)]) {
        return input[static_cast<size_t>(a)] < input[static_cast<size_t>(b)];
      }
      return a < b;
    });
    end = std::chrono::steady_clock::now();
    sort_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    second_order.clear();
    if (n > 0) second_order.push_back(0);
    for (int j = 1; j < n; ++j) {
      if (input[static_cast<size_t>(first_order[static_cast<size_t>(j)])] !=
          input[static_cast<size_t>(first_order[static_cast<size_t>(j - 1)])]) {
        second_order.push_back(j);
      }
    }
    end = std::chrono::steady_clock::now();
    unique_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    int k = static_cast<int>(second_order.size());
    if (i == 0) {
      std::printf("Percentage of unique elements: %.1f %%\n",
                  static_cast<float>(k) * 100.0f / static_cast<float>(n));
    }

    iris::DMem d_second_order(second_order.data(), sizeof(int) * second_order.size());
    iris::DMem d_first_order(first_order.data(), sizeof(int) * first_order.size());
    iris::DMem d_output(output.data(), sizeof(int) * output.size());

    const int grids = (k + NUM_THREADS - 1) / NUM_THREADS;
    const size_t gws[1] = {static_cast<size_t>(grids * NUM_THREADS)};
    const size_t lws[1] = {NUM_THREADS};

    start = std::chrono::steady_clock::now();
    iris::Task task;
    void* params[5] = {&d_second_order, &d_first_order, &d_output, &n, &k};
    int pinfo[5] = {iris_r, iris_r, iris_w, sizeof(n), sizeof(k)};
    task.kernel("remap_kernel", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 5, params, pinfo);
    task.flush_out(d_output);
    task.submit(iris_gpu, nullptr, 1);
    end = std::chrono::steady_clock::now();
    kernel_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    d_output.fetch(output.data());
    end = std::chrono::steady_clock::now();
    copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    end = std::chrono::steady_clock::now();
    dealloc_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();

  std::printf("Average offload time: %f (s)\n", offload_time * 1e-9f / repeat);
  std::printf("Average execution time of memory allocation : %f (us)\n",
              (alloc_time * 1e-3f) / repeat);
  std::printf("Average execution time of memory deallocation : %f (us)\n",
              (dealloc_time * 1e-3f) / repeat);
  std::printf("Average execution time of data copy : %f (us)\n",
              (copy_time * 1e-3f) / repeat);
  std::printf("Average execution time of Thrust sequence : %f (us)\n",
              (seq_time * 1e-3f) / repeat);
  std::printf("Average execution time of Thrust sort-by-key : %f (us)\n",
              (sort_time * 1e-3f) / repeat);
  std::printf("Average execution time of Thrust unique-by-key : %f (us)\n",
              (unique_time * 1e-3f) / repeat);
  std::printf("Average execution time of remap kernel: %f (us)\n",
              (kernel_time * 1e-3f) / repeat);

  int cs1 = 0;
  int cs2 = 0;
  for (int i = 0; i < n - 1; i++) {
    cs1 ^= output[static_cast<size_t>(i)] - output[static_cast<size_t>(i + 1)];
  }
  for (int i = 0; i < n; i++) {
    cs2 ^= output[static_cast<size_t>(i)];
  }
  std::printf("Checksum: %d %d\n", cs1, cs2);

#ifdef EXAMPLE
  for (int i = 0; i < n; i++) {
    std::printf("%d ", output[static_cast<size_t>(i)]);
  }
  std::printf("\n");
#endif

  (void)run;
  return 0;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

#ifdef EXAMPLE
  const int n = 7;
#else
  const int n = std::atoi(argv[1]);
#endif
  const int repeat = std::atoi(argv[2]);

  for (int i = 0; i < 2; i++) {
    std::printf("\nRun %d\n", i);
    eval_remap(n, repeat, i);
  }

  platform.finalize();
  return 0;
}
