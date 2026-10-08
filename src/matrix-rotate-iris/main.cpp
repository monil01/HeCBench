#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static void rotate_matrix_serial(float* matrix, int n)
{
  for (int layer = 0; layer < n / 2; ++layer) {
    int first = layer;
    int last = n - 1 - layer;
    for (int i = first; i < last; ++i) {
      int offset = i - first;
      float top = matrix[first * n + i];
      matrix[first * n + i] = matrix[(last - offset) * n + first];
      matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];
      matrix[last * n + (last - offset)] = matrix[i * n + last];
      matrix[i * n + last] = top;
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <matrix size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (n <= 0 || repeat <= 0) {
    std::fprintf(stderr, "matrix size and repeat must be positive\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const size_t total = static_cast<size_t>(n) * n;
  const size_t bytes = total * sizeof(float);
  float* serial_res = static_cast<float*>(aligned_alloc(1024, bytes));
  float* parallel_res = static_cast<float*>(aligned_alloc(1024, bytes));
  if (!serial_res || !parallel_res) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(serial_res);
    std::free(parallel_res);
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      serial_res[i * n + j] = parallel_res[i * n + j] = static_cast<float>(i * n + j);
    }
  }

  for (int i = 0; i < repeat; i++) {
    rotate_matrix_serial(serial_res, n);
  }

  iris::DMem mem_parallel(parallel_res, bytes);
  constexpr int block_size = 256;
  const size_t gws[1] = {static_cast<size_t>(((n / 2 + block_size - 1) / block_size) * block_size)};
  const size_t lws[1] = {block_size};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[2] = {&mem_parallel, const_cast<int*>(&n)};
    int pinfo[2] = {iris_rw, sizeof(n)};
    task.kernel("rotate_matrix_parallel", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 2, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_parallel);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  mem_parallel.fetch(parallel_res);

  bool ok = true;
  for (int i = 0; i < n && ok; i++) {
    for (int j = 0; j < n; j++) {
      if (serial_res[i * n + j] != parallel_res[i * n + j]) {
        ok = false;
        break;
      }
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(serial_res);
  std::free(parallel_res);
  platform.finalize();
  return ok ? 0 : 1;
}
