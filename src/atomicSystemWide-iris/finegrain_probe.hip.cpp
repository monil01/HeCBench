// Diagnostic only: this is not an IRIS port or a replacement benchmark.
// Test CPU __sync and genuine GPU system-scope addition on the same address.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
  fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

__global__ void add_probe(int* p, int loops) {
  for (int j = 0; j < loops; ++j) {
    atomicAdd_system(p, 10);
    atomicAdd_system(p + 16, 1); // GPU-only completion count, separate cache line.
  }
}

int main(int argc, char** argv) {
  setbuf(stdout, nullptr);
  if (argc != 5) {
    fprintf(stderr, "usage: %s device fine|uncached|host cpu|gpu|sequential|concurrent loops\n", argv[0]);
    return 2;
  }
  const int device = atoi(argv[1]), loops = atoi(argv[4]);
  if (loops < 1 || loops > 100) return 2;
  const bool host = !strcmp(argv[2], "host");
  const bool cpu = strcmp(argv[3], "gpu") != 0;
  const bool gpu = strcmp(argv[3], "cpu") != 0;
  const bool sequential = !strcmp(argv[3], "sequential");
  CHECK(hipSetDevice(device));
  hipDeviceProp_t prop{};
  CHECK(hipGetDeviceProperties(&prop, device));
  int host_atomic = 0;
  CHECK(hipDeviceGetAttribute(&host_atomic, hipDeviceAttributeHostNativeAtomicSupported, device));
  printf("device=%d name=%s allocation=%s mode=%s loops=%d hostNativeAtomicSupported=%d\n",
         device, prop.name, argv[2], argv[3], loops, host_atomic);
  int* p = nullptr;
  if (host) {
    CHECK(hipHostMalloc((void**)&p, 128, hipHostMallocMapped | hipHostMallocCoherent));
  } else {
    const unsigned flags = !strcmp(argv[2], "fine") ? hipDeviceMallocFinegrained : hipDeviceMallocUncached;
    CHECK(hipExtMallocWithFlags((void**)&p, 128, flags));
  }
  hipPointerAttribute_t attr{};
  CHECK(hipPointerGetAttributes(&attr, p));
  printf("pointer=%p type=%d managed=%d allocationFlags=%u CPU initialization follows\n",
         (void*)p, (int)attr.type, attr.isManaged, attr.allocationFlags);
  for (int k = 0; k < 32; ++k) p[k] = 0;
  CHECK(hipDeviceSynchronize());
  if (gpu) {
    hipLaunchKernelGGL(add_probe, dim3(64), dim3(256), 0, 0, p, loops);
    CHECK(hipGetLastError());
    if (sequential) CHECK(hipDeviceSynchronize());
  }
  int cpu_completed = 0;
  if (cpu) {
    for (int k = 0; k < 16384; ++k) {
      for (int j = 0; j < loops; ++j) {
        __sync_fetch_and_add(p, 10);
        ++cpu_completed;
      }
    }
  }
  CHECK(hipDeviceSynchronize());
  const int gpu_completed = p[16];
  const int expected_cpu = cpu ? 16384 * loops : 0;
  const int expected_gpu = gpu ? 16384 * loops : 0;
  const int expected = (expected_cpu + expected_gpu) * 10;
  const int actual = p[0];
  const bool ok = actual == expected && cpu_completed == expected_cpu && gpu_completed == expected_gpu;
  printf("actual=%d expected=%d cpu_completed=%d/%d gpu_completed=%d/%d %s\n",
         actual, expected, cpu_completed, expected_cpu, gpu_completed, expected_gpu,
         ok ? "PASS" : "FAIL");
  if (host) { CHECK(hipHostFree(p)); }
  else { CHECK(hipFree(p)); }
  return ok ? 0 : 1;
}
