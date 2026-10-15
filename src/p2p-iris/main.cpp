#include <iris/iris.hpp>
#include <hip/hip_runtime_api.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef HECBENCH_P2P_ELEMENTS
#define HECBENCH_P2P_ELEMENTS (16 * 1024 * 1024)
#endif
static void check_impl(int status, const char* operation) {
  if (status != IRIS_SUCCESS || iris_error_count()) {
    std::fprintf(stderr, "IRIS peer error %d at %s\n", status, operation); std::exit(2);
  }
}
#define check(expression) check_impl((expression), #expression)
static void dump(const char* suffix, const std::vector<float>& values) {
  const char* prefix = std::getenv("HECBENCH_P2P_DUMP");
  if (!prefix) return;
  const std::string path = std::string(prefix) + suffix;
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file || std::fwrite(values.data(), sizeof(float), values.size(), file) != values.size()) std::exit(2);
  if (std::fclose(file)) std::exit(2);
}
static bool pair_access(int first, int second, int repeat) {
  const size_t n = HECBENCH_P2P_ELEMENTS, bytes = n*sizeof(float);
  std::vector<float> a(n), b(n, -1.0f);
  for (size_t i = 0; i < n; ++i) a[i] = static_cast<float>(i % 4096);
  iris::DMem source(a.data(), bytes), destination(b.data(), bytes);
  check(iris_hip_enable_peer_access(first, second));
  check(iris_hip_enable_peer_access(second, first));
  {
    iris::Task upload_a, upload_b;
    check(upload_a.h2d(&source)); check(upload_a.submit(first, nullptr, true));
    check(upload_b.h2d(&destination)); check(upload_b.submit(second, nullptr, true));
  }
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    if (i % 2 == 0) check(iris_dmem_peer_copy(destination.mem(), second, source.mem(), first, bytes));
    else check(iris_dmem_peer_copy(source.mem(), first, destination.mem(), second, bytes));
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::printf("Peer-to-peer IRIS copy GPU%d/GPU%d: %.3f GB/s\n", first, second, repeat*bytes/seconds/1e9);
  check(destination.fetch(b.data()));
  for (size_t i = 0; i < n; ++i) if (b[i] != static_cast<float>(i % 4096)) return false;
  size_t local[] = {256}, global[] = {(n+255)/256*256};
  {
    iris::Task forward("peer_forward", false, true);
    void* args[] = {&source, &destination, const_cast<size_t*>(&n)};
    int info[] = {iris_r, iris_w, sizeof(n)};
    check(forward.kernel("SimpleKernel", 1, nullptr, global, local, 3, args, info));
    if (iris_task_bind_hip_remote_dmem(forward.task(), 1, first) != IRIS_ERROR ||
        iris_task_bind_hip_remote_dmem(forward.task(), 2, first) != IRIS_ERROR ||
        iris_task_bind_hip_remote_dmem(forward.task(), 0, second) != IRIS_ERROR) {
      std::fprintf(stderr, "FAIL invalid remote binding accepted\n"); return false;
    }
    check(iris_task_bind_hip_remote_dmem(forward.task(), 0, first));
    check(forward.submit(second, nullptr, true));
    if (iris_task_bind_hip_remote_dmem(forward.task(), 0, first) != IRIS_ERROR) {
      std::fprintf(stderr, "FAIL submitted task remote rebind accepted\n"); return false;
    }
    check(iris_task_release(forward.task()));
  }
  {
    iris::Task reverse;
    void* args[] = {&destination, &source, const_cast<size_t*>(&n)};
    int info[] = {iris_r, iris_w, sizeof(n)};
    check(reverse.kernel("SimpleKernel", 1, nullptr, global, local, 3, args, info));
    check(iris_task_bind_hip_remote_dmem(reverse.task(), 0, second));
    check(reverse.submit(first, nullptr, true));
  }
  // Prove these were remote reads, not IRIS migration followed by local reads.
  if (iris_dmem_is_allocated_on_device(source.mem(), second) != 0 ||
      iris_dmem_is_allocated_on_device(destination.mem(), first) != 0) {
    std::fprintf(stderr, "FAIL unexpected peer input migration\n"); return false;
  }
  check(source.fetch(a.data())); check(destination.fetch(b.data()));
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != static_cast<float>((i % 4096)*4) || b[i] != static_cast<float>((i % 4096)*2)) {
      std::fprintf(stderr, "FAIL peer output at %zu\n", i); return false;
    }
  }
  const std::string pair = "." + std::to_string(first) + "_" + std::to_string(second);
  dump((pair + ".g0.bin").c_str(), a); dump((pair + ".g1.bin").c_str(), b);
  std::printf("PASS GPU%d/GPU%d: %zu peer-copy values and %zu remote-kernel outputs; no input migration\n", first, second, n, 2*n);
  return true;
}
int main(int argc, char** argv) {
  if (argc != 2 || std::atoi(argv[1]) <= 0 || HECBENCH_P2P_ELEMENTS <= 0) {
    std::printf("Usage: %s <repeat>\n", argv[0]); return 1;
  }
  // Must precede the first HIP/HSA initialization; the Makefile also exports it.
  setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 0);
  setenv("IRIS_ARCHS", "HIP", 1);
  iris::Platform platform;
  check(platform.init(&argc, &argv, 1));
  check(iris_init_devices(1));
  int devices = 0;
  check(iris_device_count(&devices));
  if (devices < 2) { std::fprintf(stderr, "Two HIP GPUs required; not a PASS\n"); return 2; }
  bool ok = true;
  int pairs = 0;
  for (int first = 0; first < devices; ++first) for (int second = first+1; second < devices; ++second) {
    int supported = 0;
    if (hipDeviceCanAccessPeer(&supported, first, second) != hipSuccess) return 2;
    if (!supported) continue;
    ++pairs;
    ok = pair_access(first, second, std::atoi(argv[1])) && ok;
  }
  check(platform.finalize());
  return pairs && ok && !iris_error_count() ? 0 : 1;
}
