#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

typedef unsigned int T;

static bool verifySort(const T* keys, const size_t size) {
  bool passed = true;
  for (size_t i = 0; i < size - 1; i++) {
    if (keys[i] > keys[i + 1]) {
      passed = false;
#ifdef VERBOSE_OUTPUT
      std::cout << "Idx: " << i;
      std::cout << " Key: " << keys[i] << "\n";
#endif
      break;
    }
  }
  std::cout << (passed ? "PASS" : "FAIL") << std::endl;
  return passed;
}

static void* checkedMalloc(size_t bytes) {
  void* ptr = std::malloc(bytes);
  if (!ptr) {
    std::fprintf(stderr, "host allocation failed\n");
    std::exit(1);
  }
  return ptr;
}

static void launch_reduce(iris::DMem& mem_in, iris::DMem& mem_isums,
                          size_t size, unsigned int shift,
                          size_t global_wsize, size_t local_wsize) {
  size_t gws[1] = {global_wsize};
  size_t lws[1] = {local_wsize};
  void* params[4] = {&mem_in, &mem_isums, &size, &shift};
  int pinfo[4] = {iris_r, iris_w, sizeof(size), sizeof(shift)};
  iris::Task task;
  task.kernel("reduce", 1, nullptr, gws, lws, 4, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_top_scan(iris::DMem& mem_isums, size_t num_work_groups,
                            size_t local_wsize) {
  size_t gws[1] = {local_wsize};
  size_t lws[1] = {local_wsize};
  void* params[2] = {&mem_isums, &num_work_groups};
  int pinfo[2] = {iris_rw, sizeof(num_work_groups)};
  iris::Task task;
  task.kernel("top_scan", 1, nullptr, gws, lws, 2, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_bottom_scan(iris::DMem& mem_out, iris::DMem& mem_in,
                               iris::DMem& mem_isums, size_t size,
                               unsigned int shift, size_t global_wsize,
                               size_t local_wsize) {
  size_t gws[1] = {global_wsize};
  size_t lws[1] = {local_wsize};
  void* params[5] = {&mem_out, &mem_in, &mem_isums, &size, &shift};
  int pinfo[5] = {iris_w, iris_r, iris_r, sizeof(size), sizeof(shift)};
  iris::Task task;
  task.kernel("bottom_scan", 1, nullptr, gws, lws, 5, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("Usage: %s <problem size> <number of passes>\n.", argv[0]);
    return -1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int select = std::atoi(argv[1]);
  int passes = std::atoi(argv[2]);

  int probSizes[4] = {1, 8, 32, 64};
  size_t size = probSizes[select];
  size = (size * 1024 * 1024) / sizeof(T);

  size_t bytes = size * sizeof(T);
  T* h_idata = static_cast<T*>(checkedMalloc(bytes));
  T* h_odata = static_cast<T*>(checkedMalloc(bytes));

  std::cout << "Initializing host memory." << std::endl;
  for (size_t i = 0; i < size; i++) {
    h_idata[i] = i % 16;
    h_odata[i] = size - i;
  }

  std::cout << "Running benchmark with input array length " << size
            << std::endl;

  const size_t local_wsize = 256;
  const size_t global_wsize = 16384;
  const size_t num_work_groups = global_wsize / local_wsize;
  const int radix_width = 4;
  const int num_digits = 16;

  T* h_isums = static_cast<T*>(
      checkedMalloc(num_work_groups * num_digits * sizeof(T)));

  iris::DMem mem_idata(h_idata, bytes);
  iris::DMem mem_odata(h_odata, bytes);
  iris::DMem mem_isums(h_isums, num_work_groups * num_digits * sizeof(T));

  int final_buffer = 0;
  double time = 0.0;
  for (int k = 0; k < passes; k++) {
    auto start = std::chrono::steady_clock::now();
    for (unsigned int shift = 0; shift < sizeof(T) * 8; shift += radix_width) {
      bool even = ((shift / radix_width) % 2 == 0) ? true : false;
      iris::DMem& mem_in = even ? mem_idata : mem_odata;
      iris::DMem& mem_out = even ? mem_odata : mem_idata;
      final_buffer = even ? 1 : 0;

      launch_reduce(mem_in, mem_isums, size, shift, global_wsize, local_wsize);
      launch_top_scan(mem_isums, num_work_groups, local_wsize);
      launch_bottom_scan(mem_out, mem_in, mem_isums, size, shift, global_wsize,
                         local_wsize);
    }
    auto end = std::chrono::steady_clock::now();
    time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
  }

  std::printf("Average elapsed time of sort: %lf (s)\n",
              time * 1e-9 / passes);

  if (final_buffer == 0) {
    mem_idata.fetch(h_odata);
  } else {
    mem_odata.fetch(h_odata);
  }
  bool ok = verifySort(h_odata, size);

  T* h_reference = static_cast<T*>(checkedMalloc(bytes));
  time = 0.0;
  for (int k = 0; k < passes; k++) {
    std::copy(h_idata, h_idata + size, h_reference);
    auto start = std::chrono::steady_clock::now();
    std::sort(h_reference, h_reference + size);
    auto end = std::chrono::steady_clock::now();
    time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
  }
  std::printf("Average elapsed time of std::sort reference: %lf (s)\n",
              time * 1e-9 / passes);
  ok = verifySort(h_reference, size) && ok;

  std::free(h_idata);
  std::free(h_odata);
  std::free(h_isums);
  std::free(h_reference);

  platform.finalize();
  return ok ? 0 : 1;
}
