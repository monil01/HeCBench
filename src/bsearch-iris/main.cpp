#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifndef Real_t
#define Real_t float
#endif

#ifdef DEBUG
static bool verify(const Real_t* a,
                   const Real_t* z,
                   size_t* r,
                   size_t aSize,
                   size_t zSize,
                   const char* msg)
{
  bool ok = true;
  for (size_t i = 0; i < zSize; ++i) {
    if (!(r[i] + 1 < aSize && a[r[i]] <= z[i] && z[i] < a[r[i] + 1])) {
      std::printf("%s: incorrect result:\n", msg);
      std::printf("index = %zu r[index] = %zu\n", i, r[i]);
      std::printf("%f <= %f < %f\n",
                  static_cast<double>(a[r[i]]),
                  static_cast<double>(z[i]),
                  static_cast<double>(a[r[i] + 1]));
      ok = false;
      break;
    }
    r[i] = 0xFFFFFFFF;
  }
  return ok;
}
#endif

static bool run_kernel(const char* kernel_name,
                       const char* label,
                       size_t zSize,
                       size_t n,
                       int repeat,
                       iris::DMem& mem_a,
                       iris::DMem& mem_z,
                       iris::DMem& mem_r,
                       const Real_t* a,
                       const Real_t* z,
                       size_t* r,
                       size_t aSize)
{
  const size_t block_size = 256;
  const size_t gws[1] = { ((zSize + block_size - 1) / block_size) * block_size };
  const size_t lws[1] = { block_size };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[5] = { &mem_a, &mem_z, &mem_r, &zSize, &n };
    int pinfo[5] = { iris_r, iris_r, iris_w, sizeof(zSize), sizeof(n) };
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_r);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time (%s) %f (s)\n",
              label,
              (time * 1e-9f) / repeat);

#ifdef DEBUG
  mem_r.fetch(r);
  return verify(a, z, r, aSize, zSize, label);
#else
  (void)a;
  (void)z;
  (void)r;
  (void)aSize;
  return true;
#endif
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage ./main <number of elements> <repeat>\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const size_t numElem = std::atol(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (numElem < 2 || repeat <= 0) {
    std::fprintf(stderr, "number of elements must be at least 2 and repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  std::srand(2);
  const size_t aSize = numElem;
  const size_t zSize = 2 * aSize;
  Real_t* a = nullptr;
  Real_t* z = nullptr;
  size_t* r = nullptr;
  if (posix_memalign(reinterpret_cast<void**>(&a), 1024, aSize * sizeof(Real_t)) ||
      posix_memalign(reinterpret_cast<void**>(&z), 1024, zSize * sizeof(Real_t)) ||
      posix_memalign(reinterpret_cast<void**>(&r), 1024, zSize * sizeof(size_t))) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(a);
    std::free(z);
    std::free(r);
    platform.finalize();
    return 1;
  }

  const size_t n = aSize - 1;
  for (size_t i = 0; i < aSize; i++) a[i] = static_cast<Real_t>(i);
  for (size_t i = 0; i < zSize; i++) z[i] = static_cast<Real_t>(std::rand() % n);

  iris::DMem mem_a(a, aSize * sizeof(Real_t));
  iris::DMem mem_z(z, zSize * sizeof(Real_t));
  iris::DMem mem_r(r, zSize * sizeof(size_t));

  bool ok = true;
  ok = run_kernel("kernel_BS", "bs1", zSize, n, repeat, mem_a, mem_z, mem_r, a, z, r, aSize) && ok;
  ok = run_kernel("kernel_BS2", "bs2", zSize, n, repeat, mem_a, mem_z, mem_r, a, z, r, aSize) && ok;
  ok = run_kernel("kernel_BS3", "bs3", zSize, n, repeat, mem_a, mem_z, mem_r, a, z, r, aSize) && ok;
  ok = run_kernel("kernel_BS4", "bs4", zSize, n, repeat, mem_a, mem_z, mem_r, a, z, r, aSize) && ok;

#ifdef DEBUG
  std::printf("%s\n", ok ? "PASS" : "FAIL");
#endif

  std::free(a);
  std::free(z);
  std::free(r);
  platform.finalize();
  return ok ? 0 : 1;
}
