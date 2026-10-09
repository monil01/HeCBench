#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define VECTOR_SIZE (1024*1024)
#define BLOCK_SIZE 256
#define TOTAL_ITERATIONS (1024)

static bool shmembenchGPU(double *c, const long size, const int repeat) {
  const int TOTAL_BLOCKS = size / BLOCK_SIZE;
  iris::DMem cd(c, (size_t)size * sizeof(double));

  size_t lws[1] = {BLOCK_SIZE};
  size_t gws[1] = {(size_t)(TOTAL_BLOCKS / 4) * BLOCK_SIZE};

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[1] = {&cd};
    int pinfo[1] = {iris_rw};
    task.kernel("benchmark_shmem", 1, nullptr, gws, lws, 1, params, pinfo);
    if (i == repeat - 1) task.flush_out(cd);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto time_shmem_128b = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)repeat;
  std::printf("Average kernel execution time : %f (ms)\n", time_shmem_128b * 1e-6);

  double sum = 0;
  for (long i = 0; i < size; i++) sum += c[i];
  const bool ok = (sum == 21256458760384741137729978368.00);
  if (!ok)
    std::printf("checksum failed\n");

  std::printf("Memory throughput\n");
  const long long operations_bytes  = (6LL+4*5*TOTAL_ITERATIONS+6)*size*sizeof(float);
  const long long operations_128bit = (6LL+4*5*TOTAL_ITERATIONS+6)*size/4;

  std::printf("\tusing 128bit operations : %8.2f GB/sec (%6.2f billion accesses/sec)\n",
    (double)operations_bytes / time_shmem_128b,
    (double)operations_128bit / time_shmem_128b);
  return ok;
}

int main(int argc, char* argv[]) {
  std::printf("Shared memory bandwidth microbenchmark\n");

  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int n = std::atoi(argv[1]);
  unsigned int datasize = VECTOR_SIZE * sizeof(double);

  std::printf("Buffer sizes: %dMB\n", datasize/(1024*1024));

  double *c = (double*)std::malloc(datasize);
  if (!c) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }
  std::memset(c, 0, datasize);

  bool ok = shmembenchGPU(c, VECTOR_SIZE, n);

  std::free(c);
  platform.finalize();

  return ok ? 0 : 2;
}
