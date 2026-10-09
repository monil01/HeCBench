#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

static bool run_benchmark(
    const char* kernel_name,
    int grid_size,
    int block_size,
    int arrayLength,
    int repeat,
    float GB,
    int checksum,
    iris::DMem& mem_in,
    iris::DMem& mem_out,
    int* out)
{
  std::chrono::high_resolution_clock::time_point t1, t2;
  t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    *out = 0;
    mem_out.update(out);

    size_t gws[1] = {(size_t)grid_size * (size_t)block_size};
    size_t lws[1] = {(size_t)block_size};
    iris::Task task;
    void* params[3] = {&mem_in, &mem_out, &arrayLength};
    int pinfo[3] = {iris_r, iris_rw, (int)sizeof(arrayLength)};
    task.kernel(kernel_name, 1, nullptr, gws, lws, 3, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  t2 = std::chrono::high_resolution_clock::now();
  double times = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  std::cout << "Thread block size: " << block_size << ", ";
  std::cout << "The average performance of reduction is " << 1.0E-09 * GB / times << " GBytes/sec" << std::endl;
  const int sum = *out;
  const bool ok = (sum == checksum);
  if (ok) {
    std::cout << "VERIFICATION: PASS" << std::endl << std::endl;
  } else {
    std::cout << "VERIFICATION: FAIL!!" << std::endl << std::endl;
  }
  return ok;
}

int main(int argc, char** argv)
{
  int arrayLength = 52428800;
  int block_sizes[] = {128, 256, 512, 1024};
  int N = 100;

  if (argc == 3) {
    arrayLength = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::cout << "Array size: " << arrayLength * sizeof(int) / 1024.0 / 1024.0 << " MB" << std::endl;
  std::cout << "Repeat the kernel execution: " << N << " times" << std::endl;

  std::vector<int> array((size_t)arrayLength);
  int checksum = 0;
  for (int i = 0; i < arrayLength; i++) {
    array[i] = std::rand() % 2;
    checksum += array[i];
  }

  std::cout << "Device name: IRIS HIP backend" << std::endl;

  int out = 0;
  iris::DMem mem_in(array.data(), sizeof(int) * (size_t)arrayLength);
  iris::DMem mem_out(&out, sizeof(int));

  for (int i = 0; i < N; i++) {
    out = 0;
    mem_out.update(&out);
    size_t gws[1] = {2048u * 256u};
    size_t lws[1] = {256u};
    iris::Task task;
    void* params[3] = {&mem_in, &mem_out, &arrayLength};
    int pinfo[3] = {iris_r, iris_rw, (int)sizeof(arrayLength)};
    task.kernel("atomic_reduction", 1, nullptr, gws, lws, 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  float GB = (float)arrayLength * sizeof(int) * N;
  bool all_ok = true;

  for (size_t k = 0; k < sizeof(block_sizes) / sizeof(int); k++) {
    int block_size = block_sizes[k];
    int blocks = std::min((arrayLength + block_size - 1) / block_size, 2048);
    all_ok &= run_benchmark("atomic_reduction", blocks, block_size, arrayLength, N, GB, checksum, mem_in, mem_out, &out);
    all_ok &= run_benchmark("atomic_reduction_v2", blocks / 2, block_size, arrayLength, N, GB, checksum, mem_in, mem_out, &out);
    all_ok &= run_benchmark("atomic_reduction_v4", blocks / 4, block_size, arrayLength, N, GB, checksum, mem_in, mem_out, &out);
    all_ok &= run_benchmark("atomic_reduction_v8", blocks / 8, block_size, arrayLength, N, GB, checksum, mem_in, mem_out, &out);
    all_ok &= run_benchmark("atomic_reduction_v16", blocks / 16, block_size, arrayLength, N, GB, checksum, mem_in, mem_out, &out);
  }

  platform.finalize();
  return all_ok ? 0 : 2;
}
