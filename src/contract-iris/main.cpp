#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define GPU_NUM_THREADS 256

static constexpr int nContractions = 18;

static int rounded_division(int number1, int number2)
{
  if (number1 % number2 == 0) return number1 / number2;
  return number1 / number2 + 1;
}

template <typename T>
static int contract(const char* kernel_name, const int max_N, const int max_C, const int repeat)
{
  const size_t tensor_size = static_cast<size_t>(max_N) * max_N * max_N * max_C;
  const size_t adj_size = static_cast<size_t>(max_N) * max_N;
  const int output_size = max_N * max_N * max_C * nContractions;

  std::vector<T> tensor_value(tensor_size, static_cast<T>(1));
  std::vector<T> adj_value(adj_size, static_cast<T>(1));
  std::vector<T> value(output_size);

  iris::DMem mem_tensor(tensor_value.data(), tensor_value.size() * sizeof(T));
  iris::DMem mem_adj(adj_value.data(), adj_value.size() * sizeof(T));
  iris::DMem mem_value(value.data(), value.size() * sizeof(T));

  const size_t gws[1] = {
    static_cast<size_t>(rounded_division(output_size, GPU_NUM_THREADS) * GPU_NUM_THREADS)
  };
  const size_t lws[1] = { GPU_NUM_THREADS };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[6] = {
      &mem_tensor,
      &mem_adj,
      &mem_value,
      const_cast<int*>(&output_size),
      const_cast<int*>(&max_N),
      const_cast<int*>(&max_C)
    };
    int pinfo[6] = {
      iris_r,
      iris_r,
      iris_w,
      sizeof(output_size),
      sizeof(max_N),
      sizeof(max_C)
    };
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 6, params, pinfo);
    task.flush_out(mem_value);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  double checksum = 0.0;
  for (T v : value) checksum += static_cast<double>(v);
  const auto minmax = std::minmax_element(value.begin(), value.end());
  std::printf("Checksum: %lf min:%lf max:%lf\n",
              checksum,
              static_cast<double>(*minmax.first),
              static_cast<double>(*minmax.second));
  return 0;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <dimension> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int max_N = std::atoi(argv[1]);
  const int max_C = nContractions;
  const int repeat = std::atoi(argv[2]);

  contract<float>("contraction_float", max_N, max_C, repeat);
  contract<double>("contraction_double", max_N, max_C, repeat);

  platform.finalize();
  return 0;
}
