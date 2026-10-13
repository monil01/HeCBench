#include <iris/iris.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include "verification.h"

static void check(int result) {
  if (result != IRIS_SUCCESS) throw std::runtime_error("IRIS API failure " + std::to_string(result));
}

int main(int argc, char** argv) {
  try {
    size_t bound = 100000001;
    if (argc == 3 && std::string(argv[1]) == "--max-size") {
      size_t consumed = 0;
      bound = std::stoull(argv[2], &consumed);
      if (consumed != std::string(argv[2]).size() || bound < 2 || bound > 100000001)
        throw std::runtime_error("max-size must be in [2,100000001]");
    } else if (argc != 1) throw std::runtime_error("usage: main [--max-size N]");
    std::vector<size_t> sizes{2,123,3411,83621,872163,100000001};
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(), [bound](size_t n){return n > bound;}), sizes.end());
    iris::Platform platform;
    check(platform.init(&argc, &argv, true));
    int count = 0, device = -1;
    check(iris_device_count(&count));
    for (int i = 0; i < count; ++i) {
      int backend = -1, type = 0; char name[256] = {};
      check(iris_device_info(i, iris_backend, &backend, nullptr));
      check(iris_device_info(i, iris_type, &type, nullptr));
      check(iris_device_info(i, iris_name, name, nullptr));
      printf("IRIS device %d: %s backend=%d type=%d\n", i, name, backend, type);
      if (backend == iris_hip && (type & iris_gpu) && device < 0) device = i;
    }
    if (device < 0) throw std::runtime_error("no IRIS HIP GPU");
    printf("Selected IRIS HIP GPU device=%d\n", device);
    std::vector<int32_t> source(sizes.back()), input(sizes.back());
    std::srand(0);
    for (auto& v : source) v = std::abs(int32_t(std::rand())) % std::numeric_limits<int32_t>::max();
    bool all_ok = true;
    for (size_t n : sizes) {
      for (size_t i=0; i<n; ++i) input[i] = (source[i] & ((1U<<13)-1)) + 87231;
      printf("Size = %10zu\n", n);
      auto offload_start = std::chrono::steady_clock::now();
      size_t blocks = std::min(size_t(4096), (n+255)/256);
      size_t words = ((13*n)/64+1)*2;
      std::vector<uint32_t> output(words, 0);
      std::vector<int32_t> lows(blocks), highs(blocks);
      int32_t minimum = 0; unsigned char bits = 0;
      {
        iris::DMem in(input.data(), n*sizeof(int32_t)), out(output.data(), words*sizeof(uint32_t));
        iris::DMem low(lows.data(), blocks*sizeof(int32_t)), high(highs.data(), blocks*sizeof(int32_t));
        iris::DMem minmem(&minimum, sizeof(minimum)), bitmem(&bits, sizeof(bits));
        size_t gws = blocks*256, lws = 256, final_gws = 256;
        auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < 1000; ++iteration) {
          iris::Task scan_task, config_task, pack_task;
          void* scan_args[] = {&low,&high,&in,&n};
          int scan_info[] = {iris_w,iris_w,iris_r,sizeof(n)};
          check(scan_task.kernel("scan",1,nullptr,&gws,&lws,4,scan_args,scan_info));
          check(scan_task.submit(device,nullptr,true));
          void* config_args[] = {&low,&high,&bitmem,&minmem,&blocks};
          int config_info[] = {iris_r,iris_r,iris_w,iris_w,sizeof(blocks)};
          check(config_task.kernel("config",1,nullptr,&final_gws,&lws,5,config_args,config_info));
          check(config_task.submit(device,nullptr,true));
          void* pack_args[] = {&bitmem,&minmem,&out,&in,&n,&words};
          int pack_info[] = {iris_r,iris_r,iris_w,iris_r,sizeof(n),sizeof(words)};
          check(pack_task.kernel("pack",1,nullptr,&gws,&lws,6,pack_args,pack_info));
          check(pack_task.submit(device,nullptr,true));
        }
        printf("Total kernel execution time (1000 iterations) = %f (s)\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
        iris::Task flush;
        check(flush.flush_out(out)); check(flush.flush_out(minmem)); check(flush.flush_out(bitmem));
        check(flush.submit(device,nullptr,true));
      }
      printf("Device offload time = %f (s)\n",
             std::chrono::duration<double>(std::chrono::steady_clock::now()-offload_start).count());
      all_ok &= verify(source,input,output.data(),n,bits,minimum);
    }
    check(platform.finalize());
    return all_ok ? 0 : 1;
  } catch (const std::exception& error) { fprintf(stderr,"%s\n",error.what()); return 1; }
}
