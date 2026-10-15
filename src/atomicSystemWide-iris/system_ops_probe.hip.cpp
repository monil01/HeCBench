// Diagnostic, not a completed IRIS benchmark. Isolate each system atomic.
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
  fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

__device__ int apply_value(int old, int v, int op) {
  if (op == 8) return old & v;
  if (op == 9) return old | v;
  if (op == 10) return old ^ v;
  if (op == 11) return old > v ? old : v;
  return old < v ? old : v;
}

__global__ void probe(int* p, int op, int loops, int threads) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= threads) return;
  for (int j = 0; j < loops; ++j) {
    const int v = op == 5 || op == 8 ? 2 * tid + 7 :
                  op == 6 || op == 9 ? int(1u << (unsigned(tid) & 31)) : tid;
    if (op == 0) atomicAdd_system(p, 10);
    else if (op == 1) atomicExch_system(p, tid);
    else if (op == 2) atomicMax_system(p, tid);
    else if (op == 3) atomicMin_system(p, tid);
    else if (op == 4) atomicCAS_system(p, tid - 1, tid);
    else if (op == 5) atomicAnd_system(p, v);
    else if (op == 6) atomicOr_system(p, v);
    else if (op == 7) atomicXor_system(p, tid);
    else {
      int old = atomicCAS_system(p, 0, 0);
      bool done = false;
      // Bounded retries: report failure rather than creating an infinite kernel.
      for (int tries = 0; tries < 4096 && !done; ++tries) {
        const int expected = old;
        old = atomicCAS_system(p, expected, apply_value(expected, v, op));
        done = old == expected;
      }
      if (!done) atomicAdd_system(p + 32, 1);
    }
    atomicAdd_system(p + 16, 1);
  }
}

int main(int argc, char** argv) {
  setbuf(stdout, nullptr);
  if (argc != 7) {
    fprintf(stderr, "usage: %s device fine|host op[0..12] threads loops gpu|concurrent\n", argv[0]);
    return 2;
  }
  const int device = atoi(argv[1]), op = atoi(argv[3]);
  const int threads = atoi(argv[4]), loops = atoi(argv[5]);
  const bool host = !strcmp(argv[2], "host"), cpu = !strcmp(argv[6], "concurrent");
  if (op < 0 || op > 12 || threads < 1 || threads > 16384 || loops < 1 || loops > 10) return 2;
  CHECK(hipSetDevice(device));
  int* p = nullptr;
  if (host) { CHECK(hipHostMalloc((void**)&p, 192, hipHostMallocMapped | hipHostMallocCoherent)); }
  else { CHECK(hipExtMallocWithFlags((void**)&p, 192, hipDeviceMallocFinegrained)); }
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  hsa_status_t hs = hsa_amd_pointer_info(p, &info, nullptr, nullptr, nullptr);
  printf("device=%d kind=%s op=%d threads=%d loops=%d mode=%s pointer=%p hsa_status=%d global_flags=%u\n",
         device, argv[2], op, threads, loops, argv[6], (void*)p, int(hs), info.global_flags);
  if (hs != HSA_STATUS_SUCCESS) return 2;
  for (int k = 0; k < 48; ++k) p[k] = 0;
  p[0] = op == 5 || op == 7 || op == 8 || op == 10 ? 255 : 0;
  CHECK(hipDeviceSynchronize());
  hipLaunchKernelGGL(probe, dim3((threads + 255) / 256), dim3(256), 0, 0, p, op, loops, threads);
  CHECK(hipGetLastError());
  int cpu_completed = 0;
  if (cpu) {
    for (int i = threads; i < 2 * threads; ++i) {
      for (int j = 0; j < loops; ++j) {
        const int v = op == 5 || op == 8 ? 2 * i + 7 :
                      op == 6 || op == 9 ? int(1u << (unsigned(i) & 31)) : i;
        if (op == 0) __sync_fetch_and_add(p, 10);
        else if (op == 1) __sync_lock_test_and_set(p, i);
        else if (op == 4) __sync_val_compare_and_swap(p, i - 1, i);
        else if (op == 5 || op == 8) __sync_fetch_and_and(p, v);
        else if (op == 6 || op == 9) __sync_fetch_and_or(p, v);
        else if (op == 7 || op == 10) __sync_fetch_and_xor(p, i);
        else {
          int old = __sync_val_compare_and_swap(p, 0, 0), expected;
          do {
            expected = old;
            const bool is_max = op == 2 || op == 11;
            const int next = is_max ? (old > i ? old : i) : (old < i ? old : i);
            old = __sync_val_compare_and_swap(p, expected, next);
          } while (old != expected);
        }
        ++cpu_completed;
      }
    }
  }
  CHECK(hipDeviceSynchronize());
  const int n = threads * (cpu ? 2 : 1);
  int expected = 0;
  bool range = op == 1 || op == 4;
  if (op == 0) expected = n * loops * 10;
  else if (op == 2 || op == 11) expected = n - 1;
  else if (op == 5 || op == 8) {
    expected = 255;
    for (int i = 0; i < n; ++i) expected &= 2 * i + 7;
  } else if (op == 6 || op == 9) {
    for (int i = 0; i < n; ++i) expected |= int(1u << (unsigned(i) & 31));
  } else if (op == 7 || op == 10) {
    expected = 255;
    for (int j = 0; j < loops; ++j) for (int i = 0; i < n; ++i) expected ^= i;
  }
  const bool value_ok = range ? p[0] >= 0 && p[0] < n : p[0] == expected;
  const bool ok = value_ok && p[16] == threads * loops && p[32] == 0 &&
                  cpu_completed == (cpu ? threads * loops : 0);
  printf("value=%d expected=%d range=%d value_ok=%d gpu_completed=%d/%d cpu_completed=%d cas_exhausted=%d %s\n",
         p[0], expected, range, value_ok, p[16], threads * loops, cpu_completed, p[32], ok ? "PASS" : "FAIL");
  if (host) { CHECK(hipHostFree(p)); } else { CHECK(hipFree(p)); }
  return ok ? 0 : 1;
}
