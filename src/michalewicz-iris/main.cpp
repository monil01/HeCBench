#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void Error(float value, int dim)
{
  std::printf("Global minima = %f\n", value);
  float trueMin = 0.0f;
  if (dim == 2) {
    trueMin = -1.8013f;
  } else if (dim == 5) {
    trueMin = -4.687658f;
  } else if (dim == 10) {
    trueMin = -9.66015f;
  }
  std::printf("Error = %f\n", std::fabs(trueMin - value));
}

bool parse_size(const char* text, size_t* value)
{
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || parsed == 0) {
    return false;
  }
  *value = static_cast<size_t>(parsed);
  return true;
}

bool parse_int(const char* text, int* value)
{
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

} // namespace

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of vectors> <repeat>\n", argv[0]);
    return 1;
  }

  size_t n = 0;
  int repeat = 0;
  if (!parse_size(argv[1], &n) || !parse_int(argv[2], &repeat)) {
    std::fprintf(stderr, "arguments must be positive integers\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.0f, 4.0f);

  const int dims[] = {2, 5, 10};

  for (int d = 0; d < 3; d++) {
    const int dim = dims[d];
    const size_t size = n * static_cast<size_t>(dim);

    std::vector<float> values(size);
    for (size_t i = 0; i < size; i++) {
      values[i] = dis(gen);
    }

    float minValue = 0.0f;
    iris::DMem mem_values(values.data(), values.size() * sizeof(float));
    iris::DMem mem_min(&minValue, sizeof(float));

    const size_t lws[1] = { 256 };
    const size_t gws[1] = { ((n + lws[0] - 1) / lws[0]) * lws[0] };

    minValue = 0.0f;
    mem_min.update(&minValue);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      iris::Task task;
      void* params[4] = { &mem_values, &mem_min, &n, const_cast<int*>(&dim) };
      int pinfo[4] = { iris_r, iris_rw, sizeof(n), sizeof(dim) };
      task.kernel("eval", 1, nullptr, const_cast<size_t*>(gws),
                  const_cast<size_t*>(lws), 4, params, pinfo);
      task.flush_out(mem_min);
      task.submit(iris_gpu, nullptr, 1);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average execution time of kernel (dim = %d): %f (us)\n",
                dim, (time * 1e-3f) / repeat);

    mem_min.fetch(&minValue);
    Error(minValue, dim);
  }

  platform.finalize();
  return 0;
}
