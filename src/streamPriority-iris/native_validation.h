#pragma once
#include <string>
#ifndef HECBENCH_STREAM_PRIORITY_N
#define HECBENCH_STREAM_PRIORITY_N (1 << 24)
#endif
static void native_priority_dump(const char* mode, int n, const float* act,
                                  const float* grad, const float* weights,
                                  const float* prefetch) {
#ifdef HECBENCH_STREAM_PRIORITY_VALIDATE
  const char* prefix = std::getenv("HECBENCH_STREAM_PRIORITY_DUMP");
  if (!prefix) return;
#ifdef __HIPCC__
  GPU_CHECK(hipDeviceSynchronize());
#else
  GPU_CHECK(cudaDeviceSynchronize());
#endif
  const std::string path = std::string(prefix) + "." + mode + ".bin";
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) std::exit(2);
  std::vector<float> host(n);
  for (const float* pointer : {act, grad, weights, prefetch}) {
#ifdef __HIPCC__
    GPU_CHECK(hipMemcpy(host.data(), pointer, n*sizeof(float), hipMemcpyDeviceToHost));
#else
    GPU_CHECK(cudaMemcpy(host.data(), pointer, n*sizeof(float), cudaMemcpyDeviceToHost));
#endif
    if (std::fwrite(host.data(), sizeof(float), n, file) != static_cast<size_t>(n)) std::exit(2);
  }
  if (std::fclose(file)) std::exit(2);
  std::printf("dumped %s: all %d output floats\n", mode, 4*n);
#else
  (void)mode; (void)n; (void)act; (void)grad; (void)weights; (void)prefetch;
#endif
}
