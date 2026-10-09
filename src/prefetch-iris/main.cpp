#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void run_add(iris::DMem& mem_x, iris::DMem& mem_y, int num_elements, int repeat, bool flush)
{
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;
  const size_t gws[1] = { static_cast<size_t>(num_blocks) * block_size };
  const size_t lws[1] = { block_size };

  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = { &num_elements, &mem_x, &mem_y };
    int pinfo[3] = { static_cast<int>(sizeof(num_elements)), iris_r, iris_rw };
    task.kernel("add", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                3, params, pinfo);
    if (flush && i == repeat - 1) task.flush_out(mem_y);
    task.submit(iris_gpu, nullptr, 1);
  }
}

static int run_case(const char* label, int num_elements, int repeat)
{
  std::printf("%s\n", label);

  std::vector<float> A(num_elements, 1.0f);
  std::vector<float> B(num_elements, 2.0f);
  iris::DMem mem_a(A.data(), static_cast<size_t>(num_elements) * sizeof(float));
  iris::DMem mem_b(B.data(), static_cast<size_t>(num_elements) * sizeof(float));

  auto start = std::chrono::steady_clock::now();
  run_add(mem_a, mem_b, num_elements, repeat, true);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time: %f (ms)\n", time * 1e-6f / repeat);

  mem_b.fetch(B.data());
  float max_error = 0.0f;
  for (int i = 0; i < num_elements; i++) {
    max_error = std::fmax(max_error, std::fabs(B[i] - (repeat + 2.0f)));
  }

  const bool ok = (max_error == 0.0f);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int num_elements = 64 * 1024 * 1024;
  int errors = 0;

  std::printf("info: set device to 0\n");
  for (int i = 0; i < 10; i++) {
    errors += run_case("Concurrent managed access with prefetch", num_elements, repeat);
  }

  for (int i = 0; i < 10; i++) {
    errors += run_case("Concurrent managed access without prefetch", num_elements, repeat);
  }

  platform.finalize();
  return errors == 0 ? 0 : 1;
}
