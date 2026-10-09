#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#define BLOCK_SIZE 256

template <typename scalar_t>
static const char* kernel_name();

template <>
const char* kernel_name<int8_t>() { return "scatter_i8"; }
template <>
const char* kernel_name<int16_t>() { return "scatter_i16"; }
template <>
const char* kernel_name<int32_t>() { return "scatter_i32"; }
template <>
const char* kernel_name<int64_t>() { return "scatter_i64"; }
template <>
const char* kernel_name<float>() { return "scatter_f32"; }
template <>
const char* kernel_name<double>() { return "scatter_f64"; }

template <typename scalar_t>
static scalar_t source_value(int64_t i)
{
  return static_cast<scalar_t>(i);
}

template <typename scalar_t>
static bool scatter(int64_t num_elems, int repeat)
{
  const size_t n = static_cast<size_t>(num_elems);
  const size_t src_size_bytes = n * sizeof(scalar_t);
  const size_t idx_size_bytes = n * sizeof(int64_t);
  const size_t out_size_bytes = n * sizeof(scalar_t);

  int64_t* h_idx = static_cast<int64_t*>(std::malloc(idx_size_bytes));
  scalar_t* h_src = static_cast<scalar_t*>(std::malloc(src_size_bytes));
  scalar_t* h_out = static_cast<scalar_t*>(std::malloc(out_size_bytes));
  if (!h_idx || !h_src || !h_out) {
    std::fprintf(stderr, "host allocation failed for %lld elements\n",
                 static_cast<long long>(num_elems));
    std::free(h_idx);
    std::free(h_src);
    std::free(h_out);
    return false;
  }

  std::srand(123);
  for (int64_t i = 0; i < num_elems; i++) {
    h_idx[i] = num_elems - 1 - i;
    h_src[i] = source_value<scalar_t>(i);
  }
  std::memset(h_out, 0, out_size_bytes);

  iris::DMem mem_idx(h_idx, idx_size_bytes);
  iris::DMem mem_src(h_src, src_size_bytes);
  iris::DMem mem_out(h_out, out_size_bytes);

  const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t gws[1] = { blocks * BLOCK_SIZE };
  size_t lws[1] = { BLOCK_SIZE };

  int64_t time = 0;
  for (int i = 0; i < repeat; i++) {
    std::memset(h_out, 0, out_size_bytes);
    mem_out.update(h_out);

    iris::Task task;
    void* params[4] = { &mem_src, &mem_idx, &mem_out, &num_elems };
    int pinfo[4] = { iris_r, iris_r, iris_w, sizeof(num_elems) };

    auto start = std::chrono::steady_clock::now();
    task.kernel(kernel_name<scalar_t>(), 1, nullptr, gws, lws, 4, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  std::printf("Average execution time of thrust::scatter: %f (us)\n",
              (time * 1e-3f) / repeat);

  mem_out.fetch(h_out);

  bool ok = true;
  for (int64_t i = 0; i < num_elems; i++) {
    if (h_out[i] != static_cast<scalar_t>(num_elems - 1 - i)) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n\n", ok ? "PASS" : "FAIL");

  std::free(h_src);
  std::free(h_idx);
  std::free(h_out);
  return ok;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t num_elements = std::atoll(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (num_elements < 0 || repeat <= 0) {
    std::fprintf(stderr, "number of elements must be non-negative and repeat must be positive\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  bool ok = true;
  std::printf("INT8 scatter\n");
  ok = scatter<int8_t>(num_elements, repeat) && ok;
  std::printf("INT16 scatter\n");
  ok = scatter<int16_t>(num_elements, repeat) && ok;
  std::printf("INT32 scatter\n");
  ok = scatter<int32_t>(num_elements, repeat) && ok;
  std::printf("INT64 scatter\n");
  ok = scatter<int64_t>(num_elements, repeat) && ok;
  std::printf("FP32 scatter\n");
  ok = scatter<float>(num_elements, repeat) && ok;
  std::printf("FP64 scatter\n");
  ok = scatter<double>(num_elements, repeat) && ok;

  platform.finalize();
  return ok ? 0 : 1;
}
