#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define BLOCK_SIZE 512

static int run_axpby(
    int total,
    int chunk_size,
    float a,
    float b,
    int arg_to_check,
    iris::DMem& mem_x,
    iris::DMem& mem_y,
    iris::DMem& mem_z,
    iris::DMem& mem_noop)
{
  int noop = 0;
  mem_noop.update(&noop);

  const size_t blocks = (static_cast<size_t>(total) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  iris::Task task;
  void* params[8] = {
    &total,
    &a,
    &b,
    &arg_to_check,
    &mem_x,
    &mem_y,
    &mem_z,
    &mem_noop
  };
  int pinfo[8] = {
    sizeof(total),
    sizeof(a),
    sizeof(b),
    sizeof(arg_to_check),
    iris_r,
    iris_r,
    iris_w,
    iris_rw
  };
  task.kernel("axpby_kernel", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 8, params, pinfo);
  task.flush_out(mem_z);
  task.flush_out(mem_noop);
  task.submit(iris_gpu, nullptr, 1);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Chunk size %8d | Total execution time of multi_tensor_axpby: %f (us)\n",
              chunk_size, time * 1e-3f);
  return 0;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <number of tensors\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int max_tensors = std::atoi(argv[1]);
  std::vector<int64_t> offsets(max_tensors + 1, 0);
  std::vector<int64_t> lengths(max_tensors, 0);

  std::srand(123);
  int64_t total64 = 0;
  for (int n = 0; n < max_tensors; n++) {
    const int64_t length = std::rand() % (1024 * 1024) + 1024;
    lengths[n] = length;
    offsets[n] = total64;
    total64 += length;
    for (int d = 0; d < 2; d++) {
      for (int64_t i = 0; i < length; i++) {
        (void)(std::rand() % length);
      }
    }
  }
  offsets[max_tensors] = total64;
  if (total64 > INT32_MAX) {
    std::fprintf(stderr, "total element count exceeds int range: %lld\n", static_cast<long long>(total64));
    platform.finalize();
    return 1;
  }

  const int total = static_cast<int>(total64);
  float* x = static_cast<float*>(std::malloc(sizeof(float) * static_cast<size_t>(total)));
  float* y = static_cast<float*>(std::malloc(sizeof(float) * static_cast<size_t>(total)));
  float* z = static_cast<float*>(std::malloc(sizeof(float) * static_cast<size_t>(total)));
  if (!x || !y || !z) {
    std::fprintf(stderr, "host allocation failed for %d elements\n", total);
    std::free(x);
    std::free(y);
    std::free(z);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (int n = 0; n < max_tensors; n++) {
    const int64_t length = std::rand() % (1024 * 1024) + 1024;
    const int64_t base = offsets[n];
    for (int64_t i = 0; i < length; i++) {
      x[base + i] = static_cast<float>(std::rand() % length);
    }
    for (int64_t i = 0; i < length; i++) {
      y[base + i] = static_cast<float>(std::rand() % length);
    }
    for (int64_t i = 0; i < length; i++) {
      z[base + i] = 0.0f;
    }
  }

  const float a = 1.f;
  const float b = 1.f;
  const int arg_to_check = -1;
  int noop = 0;

  iris::DMem mem_x(x, sizeof(float) * static_cast<size_t>(total));
  iris::DMem mem_y(y, sizeof(float) * static_cast<size_t>(total));
  iris::DMem mem_z(z, sizeof(float) * static_cast<size_t>(total));
  iris::DMem mem_noop(&noop, sizeof(int));

  for (int chunk_size = 256; chunk_size <= 1024 * 1024; chunk_size *= 2) {
    run_axpby(total, chunk_size, a, b, arg_to_check, mem_x, mem_y, mem_z, mem_noop);
    mem_z.fetch(z);

    bool ok = true;
    for (int n = 0; n < max_tensors && ok; n++) {
      const int64_t base = offsets[n];
      const int64_t length = lengths[n];
      for (int64_t i = 0; i < length; i++) {
        const float ref = a * x[base + i] + b * y[base + i];
        if (std::fabs(z[base + i] - ref) > 1e-3f) {
          ok = false;
          break;
        }
      }
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
  }

  std::free(x);
  std::free(y);
  std::free(z);
  platform.finalize();
  return 0;
}
