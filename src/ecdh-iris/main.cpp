#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define P_x 5
#define P_y 1
#define MODULUS 17
#define A 2
#define BLOCK_SIZE 256

static void run_kernel(
    const char* name,
    int num_pk,
    int repeat,
    iris::DMem& mem_x,
    iris::DMem& mem_y,
    double* average_seconds)
{
  int sk = 18;
  int px = P_x;
  int py = P_y;
  unsigned int modulus = MODULUS;
  int a = A;
  const size_t blocks = (static_cast<size_t>(num_pk) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  iris::Task task;
  void* params[9] = {
    &sk,
    &px,
    &py,
    &mem_x,
    &mem_y,
    &modulus,
    &a,
    &num_pk,
    &repeat
  };
  int pinfo[9] = {
    sizeof(sk),
    sizeof(px),
    sizeof(py),
    iris_w,
    iris_w,
    sizeof(modulus),
    sizeof(a),
    sizeof(num_pk),
    sizeof(repeat)
  };
  task.kernel(name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 9, params, pinfo);
  task.flush_out(mem_x);
  task.flush_out(mem_y);
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  *average_seconds = time * 1e-9 / repeat;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <positive number of keys> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int num_pk = std::atoi(argv[1]);
  int repeat = std::atoi(argv[2]);
  std::vector<int> pk_slow_x(num_pk);
  std::vector<int> pk_slow_y(num_pk);
  std::vector<int> pk_fast_x(num_pk);
  std::vector<int> pk_fast_y(num_pk);

  iris::DMem mem_x(pk_slow_x.data(), pk_slow_x.size() * sizeof(int));
  iris::DMem mem_y(pk_slow_y.data(), pk_slow_y.size() * sizeof(int));

  double slow_seconds = 0.0;
  run_kernel("k_slow_repeat", num_pk, repeat, mem_x, mem_y, &slow_seconds);
  mem_x.fetch(pk_slow_x.data());
  mem_y.fetch(pk_slow_y.data());
  std::printf("Average time (slow kernel): %f s\n", slow_seconds);

  double fast_seconds = 0.0;
  run_kernel("k_fast_repeat", num_pk, repeat, mem_x, mem_y, &fast_seconds);
  mem_x.fetch(pk_fast_x.data());
  mem_y.fetch(pk_fast_y.data());
  std::printf("Average time (fast kernel): %f s\n", fast_seconds);

  bool fail_pk_x = std::memcmp(pk_slow_x.data(), pk_fast_x.data(), pk_slow_x.size() * sizeof(int));
  bool fail_pk_y = std::memcmp(pk_slow_y.data(), pk_fast_y.data(), pk_slow_y.size() * sizeof(int));
  std::printf("%s\n", (fail_pk_x || fail_pk_y) ? "FAIL" : "PASS");

  platform.finalize();
  return (fail_pk_x || fail_pk_y) ? 2 : 0;
}
