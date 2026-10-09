#include <iris/iris.hpp>
#include <iris/iris_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

static void copy_dmem(iris::DMem& dst, iris::DMem& src, long len,
                      int target_device, bool flush_dst) {
  const size_t block = 256;
  const size_t gws0 =
      ((static_cast<size_t>(len) + block - 1) / block) * block;
  size_t gws[] = {gws0};
  size_t lws[] = {block};
  void* params[] = {&dst, &src, &len};
  int pinfo[] = {iris_w, iris_r, sizeof(len)};

  iris::Task task;
  task.kernel("ring_copy", 1, nullptr, gws, lws, 3, params, pinfo);
  if (flush_dst) task.flush_out(dst);
  task.submit(target_device, nullptr, 1);
}

static void init_dmem(iris::DMem& dst, long len, int target_device) {
  const size_t block = 256;
  const size_t gws0 =
      ((static_cast<size_t>(len) + block - 1) / block) * block;
  size_t gws[] = {gws0};
  size_t lws[] = {block};
  void* params[] = {&dst, &len};
  int pinfo[] = {iris_w, sizeof(len)};

  iris::Task task;
  task.kernel("ring_init", 1, nullptr, gws, lws, 2, params, pinfo);
  task.submit(target_device, nullptr, 1);
}

int main(int argc, char* argv[]) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  if (argc != 4) {
    std::printf("Usage: %s <minimum copy length> <maximum copy length> <repeat>\n",
                argv[0]);
    platform.finalize();
    return 1;
  }
  const long min_len = std::atol(argv[1]);
  const long max_len = std::atol(argv[2]);
  const int repeat = std::atoi(argv[3]);
  if (min_len <= 0 || max_len < min_len || repeat <= 0) {
    std::printf("minimum, maximum, and repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  int num_devices = iris_ndevices();
  if (num_devices <= 0) num_devices = 1;
  for (int i = 0; i < num_devices; i++) {
    std::printf("Device name: IRIS HIP device %d\n", i);
  }

  if (num_devices <= 1) {
    std::printf("Warning, only one device is detected."
                "This program is supposed to execute with multiple devices.\n");
  }

  for (long len = min_len; len <= max_len; len = len * 4) {
    const size_t data_size_bytes = static_cast<size_t>(len) * sizeof(int);
    std::vector<std::vector<int>> host_ptr(num_devices,
                                           std::vector<int>(len, 0));
    std::iota(host_ptr[0].begin(), host_ptr[0].end(), 0);

    std::vector<std::unique_ptr<iris::DMem>> device_mem;
    device_mem.reserve(num_devices);
    for (int i = 0; i < num_devices; i++) {
      device_mem.emplace_back(
          std::make_unique<iris::DMem>(host_ptr[i].data(), data_size_bytes));
    }
    init_dmem(*device_mem[0], len, 0);
    std::memset(host_ptr[0].data(), 0, data_size_bytes);

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      for (int i = 0; i < num_devices; i++) {
        const int dst = (i + 1) % num_devices;
        const bool final_copy_to_zero = n == repeat - 1 && dst == 0;
        copy_dmem(*device_mem[dst], *device_mem[i], len, i,
                  final_copy_to_zero);
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    auto time_us = time * 1e-3f / repeat;
    std::printf("----------------------------------------------------------------\n");
    std::printf("Copy length = %ld\n", len);
    std::printf("Average total exchange time: %f (us)\n", time_us);
    std::printf("Average exchange time per device: %f (us)\n",
                time_us / num_devices);

    bool ok = true;
    for (long i = 0; i < len; i++) {
      if (host_ptr[0][i] != static_cast<int>(i)) {
        ok = false;
        break;
      }
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
  }

  platform.finalize();
  return 0;
}
