#pragma once
#include <cstdlib>
#include <string>
#include <vector>
#ifndef HECBENCH_P2P_ELEMENTS
#define HECBENCH_P2P_ELEMENTS (16 * 1024 * 1024)
#endif
static void native_p2p_dump(const float* first, const float* second, size_t n,
                            int first_device, int second_device) {
#ifdef HECBENCH_P2P_VALIDATE
  const char* prefix = std::getenv("HECBENCH_P2P_DUMP");
  if (!prefix) return;
  const std::string pair = std::string(prefix) + "." + std::to_string(first_device) + "_" + std::to_string(second_device);
  std::vector<float> host(n);
  const float* pointers[] = {first, second};
  const int devices[] = {first_device, second_device};
  for (int i = 0; i < 2; ++i) {
#ifdef __HIPCC__
    if (hipSetDevice(devices[i]) != hipSuccess ||
        hipMemcpy(host.data(), pointers[i], n*sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) std::exit(2);
#else
    if (cudaSetDevice(devices[i]) != cudaSuccess ||
        cudaMemcpy(host.data(), pointers[i], n*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) std::exit(2);
#endif
    const std::string path = pair + ".g" + std::to_string(i) + ".bin";
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file || std::fwrite(host.data(), sizeof(float), n, file) != n) std::exit(2);
    if (std::fclose(file)) std::exit(2);
  }
  std::printf("dumped native peer pair %d/%d: %zu output floats\n", first_device, second_device, 2*n);
#else
  (void)first; (void)second; (void)n; (void)first_device; (void)second_device;
#endif
}
