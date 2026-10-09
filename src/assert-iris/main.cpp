#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

bool launch_kernel(const char* name, size_t blocks, size_t threads, int nparams,
                   void** params, int* pinfo)
{
  size_t gws[1] = { blocks * threads };
  size_t lws[1] = { threads };
  iris::Task task;
  task.kernel(name, 1, nullptr, gws, lws, nparams, params, pinfo);
  return task.submit(iris_gpu, nullptr, 1) == IRIS_SUCCESS;
}

bool runPerf()
{
  const size_t Nblocks = 1000;
  const size_t Nthreads = 256;

  std::printf("\nLaunch kernel to evaluate the impact of assertion on performance \n");

  std::printf("Each thread in the kernel executes threadID + 1 assertions\n");
  auto start = std::chrono::steady_clock::now();
  bool ok = launch_kernel("perfKernel", Nblocks, Nthreads, 0, nullptr, nullptr);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  std::printf("Kernel time : %f\n", time.count());
  if (!ok) return false;

  std::printf("Each thread in the kernel executes threadID assertions\n");
  start = std::chrono::steady_clock::now();
  ok = launch_kernel("perfKernel2", Nblocks, Nthreads, 0, nullptr, nullptr);
  end = std::chrono::steady_clock::now();
  time = end - start;
  std::printf("Kernel time : %f\n", time.count());

  return ok;
}

bool runTest()
{
  const size_t Nblocks = 2;
  const size_t Nthreads = 32;
  int N = 60;

  std::printf("\nLaunch kernel to generate assertion failures\n");

  std::printf("\n-- Begin assert output\n\n");
  void* params[1] = { &N };
  int pinfo[1] = { sizeof(N) };
  const bool ok = launch_kernel("testKernel", Nblocks, Nthreads, 1, params, pinfo);
  std::printf("\n-- End assert output\n\n");

  if (!ok) {
    std::printf("Device assert failed as expected, HIP error message is: device-side assert triggered\n\n");
  }

  return !ok;
}

} // namespace

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const bool perfResult = runPerf();
  const bool testResult = perfResult && runTest();

  std::printf("Test assert completed, returned %s\n",
              testResult ? "OK" : "ERROR!");

  platform.finalize();
  return testResult ? EXIT_SUCCESS : EXIT_FAILURE;
}
