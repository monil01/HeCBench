#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int nBlocks = 65536;
  const int blockSize = 256;
  int failures = 0;

  for (int ds = 32; ds >= 1; ds = ds / 2) {
    int* h_d = new int[ds]();
    iris::DMem mem_d(h_d, static_cast<size_t>(ds) * sizeof(int));
    mem_d.update(h_d);

    const size_t gws[1] = {static_cast<size_t>(nBlocks) * blockSize};
    const size_t lws[1] = {static_cast<size_t>(blockSize)};

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      iris::Task task;
      void* params[2] = {&mem_d, &ds};
      int pinfo[2] = {iris_rw, sizeof(ds)};
      task.kernel("atomicAggregate_k", 1, nullptr, const_cast<size_t*>(gws),
                  const_cast<size_t*>(lws), 2, params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
    }
    iris::Task flush;
    flush.flush_out(mem_d);
    flush.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<float> time = end - start;
    std::printf("Total kernel time (%d locations): %f (s)\n", ds, time.count());

    mem_d.fetch(h_d);

    bool ok = true;
    const int expected = blockSize / ds * nBlocks * repeat;
    for (int i = 0; i < ds; i++) {
      if (h_d[i] != expected) {
        ok = false;
        break;
      }
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    failures += ok ? 0 : 1;

    delete[] h_d;
  }

  platform.finalize();
  return failures == 0 ? 0 : 1;
}
