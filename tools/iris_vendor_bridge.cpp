#include "iris_vendor_bridge.hpp"
#include <cstdio>
#include <cstdlib>

namespace {
constexpr unsigned slots = 24, bytes = 32;
struct Arguments { alignas(16) unsigned char values[slots][bytes]; };
static_assert(sizeof(Arguments) <= 1024, "IRIS parameter wrapper capacity");
int copy_arg(void* obj, int index, size_t size, const void* value) {
  if (!obj || index < 0 || static_cast<unsigned>(index) >= slots || size > bytes || !value) return -1;
  std::memcpy(static_cast<Arguments*>(obj)->values[index], value, size);
  return 0;
}
}
extern "C" {
// Exported discovery symbol. Actual dispatch is through launch_with_obj below.
void hecbench_vendor() {}
int iris_host2hip_init() { return 0; }
int iris_host2hip_finalize() { return 0; }
int iris_host2hip_init_handles(int) { return 0; }
int iris_host2hip_finalize_handles(int) { return 0; }
int iris_host2hip_kernel_with_obj(void* obj, const char* name) {
  if (!obj || !name || std::strcmp(name, "hecbench_vendor")) return -1;
  std::memset(obj, 0, sizeof(Arguments));
  return 0;
}
int iris_host2hip_setarg_with_obj(void* obj, int index, size_t size, void* value) {
  const int status = copy_arg(obj, index, size, value);
  if (status) { std::fputs("Invalid IRIS vendor scalar argument\n", stderr); std::abort(); }
  return status;
}
int iris_host2hip_setmem_with_obj(void* obj, int index, void* ptr, size_t) {
  const int status = copy_arg(obj, index, sizeof(ptr), &ptr);
  if (status) { std::fputs("Invalid IRIS vendor DMem argument\n", stderr); std::abort(); }
  return status;
}
int iris_host2hip_launch_with_obj(void* stream, void* obj, uint32_t device,
                                  int, size_t*, size_t*) {
  void* args[slots];
  for (unsigned i = 0; i < slots; ++i) args[i] = static_cast<Arguments*>(obj)->values[i];
  const auto callback = hecbench_vendor_arg<hecbench_vendor_callback>(args, 0);
  if (!callback) { std::fputs("Missing IRIS vendor callback\n", stderr); std::abort(); }
  // IRIS supplies a pointer to the selected HIP stream handle, not the handle.
  void* selected_stream = stream ? *static_cast<void**>(stream) : nullptr;
  const int status = callback(selected_stream, device & 0xffffu, args);
  if (status) {
    // The reference IRIS loader discards this function's return value. Never
    // allow a failed library call to be reported as a successful IRIS task.
    std::fprintf(stderr, "IRIS vendor callback failed: %d\n", status);
    std::abort();
  }
  return 0;
}
}
