#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>

static size_t shrRoundUp(int group_size, size_t global_size)
{
  if (global_size == 0) return group_size;
  int r = global_size % group_size;
  return (r == 0) ? global_size : global_size + group_size - r;
}

template <typename T>
static void launch_dot(
    const char* kernel_name,
    iris::DMem& mem_a,
    iris::DMem& mem_b,
    iris::DMem& mem_d,
    size_t n,
    size_t grid_size)
{
  size_t gws[1] = { grid_size * 1024 };
  size_t lws[1] = { 1024 };
  iris::Task task;
  void* params[4] = { &mem_a, &mem_b, &mem_d, &n };
  int pinfo[4] = { iris_r, iris_r, iris_w, sizeof(n) };
  task.kernel(kernel_name, 1, nullptr, gws, lws, 4, params, pinfo);
  task.flush_out(mem_d);
  task.submit(iris_gpu, nullptr, 1);
}

template <typename T>
static void run_kernel_set(
    const char* kernel_name,
    iris::DMem& mem_a,
    iris::DMem& mem_b,
    iris::DMem& mem_d,
    T* dst,
    T dst_ref,
    size_t n,
    size_t grid_size,
    int iterations)
{
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    launch_dot<T>(kernel_name, mem_a, mem_b, mem_d, n, grid_size);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / iterations);

  mem_d.fetch(dst);
  T dst_dev = 0;
  for (size_t i = 0; i < grid_size; i++) dst_dev += dst[i];
  std::printf("%s\n\n", dst_dev == dst_ref ? "PASS" : "FAIL");
}

template <typename T>
static void dot(const size_t iNumElements, const int iNumIterations)
{
  int szLocalWorkSize = 1024;
  size_t szGlobalWorkSize = shrRoundUp(szLocalWorkSize, iNumElements);

  std::printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= %d\n",
              szGlobalWorkSize, szLocalWorkSize);

  const size_t src_size = szGlobalWorkSize;
  const size_t src_size_bytes = src_size * sizeof(T);
  const size_t grid_size = shrRoundUp(1, szGlobalWorkSize / (szLocalWorkSize * 4));

  T* srcA = static_cast<T*>(std::malloc(src_size_bytes));
  T* srcB = static_cast<T*>(std::malloc(src_size_bytes));
  T* dst = static_cast<T*>(std::malloc(grid_size * sizeof(T)));
  if (!srcA || !srcB || !dst) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(srcA);
    std::free(srcB);
    std::free(dst);
    std::exit(EXIT_FAILURE);
  }

  std::mt19937 engine(19937);
  std::uniform_int_distribution<T> dis(0, 255);

  T dst_ref = 0;
  for (size_t i = 0; i < iNumElements; ++i) {
    T s[4];
    for (int k = 0; k < 4; k++) s[k] = dis(engine);
    srcB[i] = srcA[i] = s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24);
    if constexpr (std::is_same_v<T, int>) {
      for (int k = 0; k < 4; k++) {
        if (s[k] >= 128) s[k] -= 256;
      }
    }
    for (int k = 0; k < 4; k++) dst_ref += s[k] * s[k];
  }
  for (size_t i = iNumElements; i < src_size; ++i) srcA[i] = srcB[i] = 0;

  iris::DMem mem_a(srcA, src_size_bytes);
  iris::DMem mem_b(srcB, src_size_bytes);
  iris::DMem mem_d(dst, grid_size * sizeof(T));
  const size_t n = src_size / 4;

  const char* scalar_kernel = std::is_same_v<T, unsigned> ? "dot_product_uint" : "dot_product_int";
  const char* dp4a_kernel = std::is_same_v<T, unsigned> ? "dot_product2_uint" : "dot_product2_int";

  for (size_t i = 0; i < 100; i++) {
    launch_dot<T>(scalar_kernel, mem_a, mem_b, mem_d, n, grid_size);
    launch_dot<T>(dp4a_kernel, mem_a, mem_b, mem_d, n, grid_size);
  }

  run_kernel_set(scalar_kernel, mem_a, mem_b, mem_d, dst, dst_ref, n, grid_size, iNumIterations);
  run_kernel_set(dp4a_kernel, mem_a, mem_b, mem_d, dst, dst_ref, n, grid_size, iNumIterations);

  std::free(srcA);
  std::free(srcB);
  std::free(dst);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const size_t iNumElements = std::atol(argv[1]);
  const int iNumIterations = std::atoi(argv[2]);

  std::printf("------------- Data type is int32 ---------------\n");
  dot<int>(iNumElements, iNumIterations);
  std::printf("------------- Data type is uint32 ---------------\n");
  dot<unsigned>(iNumElements, iNumIterations);

  platform.finalize();
  return EXIT_SUCCESS;
}
